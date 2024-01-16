from dataclasses import dataclass, field, asdict
import json
import pathlib
from typing import Dict, Optional, Sequence, Tuple, Union, List, Any
import psutil
import torch.nn.functional as F
import torch
from transformers import Trainer, TrainingArguments, HfArgumentParser, AutoTokenizer, AutoModelForCausalLM, BertModel, BertTokenizer
from torch.distributed.elastic.multiprocessing.errors import record
import torch.nn as nn
import numpy as np
import random
import torch
from model.llm_utils import LoadWoInit
from model.dict_llm import DictLLM
from rouge_chinese import Rouge
import jieba
import bert_score
import pandas as pd
from datetime import datetime
import os
import faiss
from tqdm import tqdm
import gpustat

torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)
np.random.seed(42)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    trainer.save_state()
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def load_model_for_hf_trainer(model : nn.Module, model_dir: str):
    """Load the state dict from disk and load it to model."""
    state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    return model
    
def load_hf_model_tok(mt_path):
    tok = AutoTokenizer.from_pretrained(mt_path, trust_remote_code=True)
    with LoadWoInit():
        model = AutoModelForCausalLM.from_pretrained(mt_path, trust_remote_code=True)
    return model, tok

@dataclass
class ModelArguments:
    mt_path: str = field(default=None)
    encoder_hidden_size : int = field(default=2)
    num_table_token : int = field(default=1)
    num_encoder_head : int = field(default=2)
    num_encoder_layers : int = field(default=1)
    special_tokens_path : str = field(default=None)
    mask_strategy : str = field(default="hierarchical")
    position_strategy : str = field(default="group")
    encoder_type : str = field(default="transformer")
    mapper_type : str = field(default="linear")
    deep_fusion : bool = field(default=False)
    max_length : int = field(default=2048)

@dataclass
class DataArguments:
    train_data_path : str = field(default=None)
    eval_data_path : Optional[str] = field(default=None)


@dataclass
class TrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch")
    remove_unused_columns : bool = False
    output_dir : str = field(default="output/")
    freeze_llm : bool = field(default=False)
    from_pretrained : str = field(default=None)
    debug_mode : bool = field(default=False)
    compute_metrics : bool = field(default=True)
    
@dataclass
class DictDataCollator(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, object]:
        batch_input_text = []
        batch_dicts = []
        batch_label_text = []
        for i in instances:
            batch_input_text.append(i['input'])
            if 'data' in i:
                batch_dicts.append(i['data'])
            if 'output' in i:
                batch_label_text.append(i['output'])
        return dict(batch_input_text=batch_input_text, batch_dicts=batch_dicts, batch_label_text=batch_label_text)


class DictLLMTrainer(Trainer):
    def compute_loss(self, dllm, inputs, return_outputs=False, **kwargs): 
        outputs = dllm(**inputs, **kwargs)
        if debug_mode and local_rank == 0:
            gpustat.print_gpustat()
        return (outputs['loss'], outputs) if return_outputs else outputs['loss']
    
    def prediction_step(
        self,
        dllm: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        labels = dllm.llm.tok(inputs["batch_label_text"], padding=True, return_tensors='pt', add_special_tokens=False)['input_ids'].to(dllm.llm.model.device)
        output_ids = labels
        
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(dllm, inputs, return_outputs=True, output_attentions=True)
                loss = loss.mean().detach()
                if self.args.compute_metrics:
                    output_ids = dllm.generate(**inputs, cut_input=True, max_new_tokens=2*labels.shape[-1], synced_gpus=False).to(dllm.llm.model.device)
        return (loss, output_ids, labels)


@record
def train():
    global local_rank
    global debug_mode
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    debug_mode = training_args.debug_mode
    
    # Load Dataset
    train_dst = json.load(open(data_args.train_data_path))
    eval_dst = None
    if data_args.eval_data_path:
        eval_dst = json.load(open(data_args.eval_data_path))
    
    # Load Model
    dllm = DictLLM(**asdict(model_args))
    
    if local_rank == 0:
        gpustat.print_gpustat()
        print(f"Available Memory: {psutil.virtual_memory().available/1024/1024/1024:.2f}GB")

    if debug_mode:
        training_args.report_to = []
        training_args.run_name = 'debug_' + getattr(training_args, 'run_name', 'run')
        training_args.log_level = 'debug'
        training_args.log_level_replica = 'debug'
        training_args.num_train_epochs = 2
        training_args.eval_steps = 16
        training_args.gradient_accumulation_steps = 1
        training_args.save_strategy = 'no'
        max_train_size = 16 * training_args.per_device_train_batch_size * training_args.world_size // 2
        max_eval_size = 4 * training_args.per_device_eval_batch_size * training_args.world_size
        train_dst = train_dst[:max_train_size]
        eval_dst = eval_dst[:max_eval_size]
        rank0_print(
            "debug mode is on, some training_args may be overwrite.",
            "num_train_epochs = 2",
            f"max_train_size = {max_train_size}",
            f"max_eval_size = {max_eval_size}",
            f"eval_steps = {training_args.eval_steps}",
        )
    
    rank0_print(f"model_args: {model_args}\ndata_args: {data_args}\ntraining_args: {training_args}\ntrain_dst size: {len(train_dst)}")
    
    if eval_dst:
        rank0_print(f"eval_dst size: {len(eval_dst)}")
    
    if training_args.freeze_llm:
        # frozen llm parameter
        for param in dllm.llm.parameters():
            param.requires_grad = False
    
    if training_args.from_pretrained:
        # load pretrained parameter
        state_dict = torch.load(os.path.join(training_args.from_pretrained, "pytorch_model.bin"), map_location='cpu')
        dllm.load_state_dict(state_dict, strict=False)
    
    model, tok = dllm, dllm.llm.tok
    
    data_collator = DictDataCollator()
    data_modules = dict(train_dataset=train_dst, eval_dataset=eval_dst, data_collator=data_collator)

    def compute_metrics(EvalPrediction):
        # prepare predictions and references
        output_ids = [[i for i in ids if i != -100] for ids in EvalPrediction.predictions.tolist()]
        original_predictions = tok.batch_decode(output_ids, skip_special_tokens=True)
        original_predictions = ["-" if not p.strip() else p for p in original_predictions]
        
        label_ids = [[i for i in ids if i != -100] for ids in EvalPrediction.label_ids.tolist()]
        original_references = tok.batch_decode(label_ids, skip_special_tokens=True)
        original_references = ["--" if not p.strip() else p for p in original_references]
        
        report_metrics = {}
        
        # rouge scores
        rank0_print("calculating rouge scores")
        start_time = datetime.now()
        predictions = [' '.join(list(p)) for p in original_predictions]
        references = [' '.join(list(r)) for r in original_references]
        rouge = Rouge()
        batch_rouge_scores = rouge.get_scores(predictions, references)
        avg_rougel_p = sum([s['rouge-l']['p'] for s in batch_rouge_scores])/len(batch_rouge_scores)
        avg_rougel_r = sum([s['rouge-l']['r'] for s in batch_rouge_scores])/len(batch_rouge_scores)
        avg_rougel_f1 = sum([s['rouge-l']['f'] for s in batch_rouge_scores])/len(batch_rouge_scores)
        report_metrics.update({'rougeL-p' : avg_rougel_p, 'rougeL-r':avg_rougel_r, 'rougeL' : avg_rougel_f1})
        rank0_print("calculating rouge scores time: ", datetime.now()-start_time)
        
        # bert_scores
        rank0_print("calculating bert_scores")
        start_time = datetime.now()
        predictions = original_predictions
        references = original_references
        try:
            batch_bert_scores = bert_score.score(predictions, references, lang='zh') # p,r,f
            batch_bert_scores = [s.cpu().tolist() for s in batch_bert_scores]
            avg_bert_score_p = sum(batch_bert_scores[0]) / len(batch_bert_scores[0])
            avg_bert_score_r = sum(batch_bert_scores[1]) / len(batch_bert_scores[1])
            avg_bert_score_f1 = sum(batch_bert_scores[2]) / len(batch_bert_scores[2])
            report_metrics.update({'bert_score-p':avg_bert_score_p, 'bert_score-r':avg_bert_score_r, 'bert_score-f1':avg_bert_score_f1})
        except:
            pass

        rank0_print("calculating bert_scores time: ", datetime.now()-start_time)
        
        # bios socres
        rank0_print("calculate bios scores")
        start_time = datetime.now()
        predictions = original_predictions
        references = original_references
        eval_model_path = os.path.join(os.environ['my_models_dir'], 'bert-base-chinese')
        bios_index = faiss.read_index(os.path.join(os.environ['my_datasets_dir'], "bios_v2.2_release/CoreData/TermDiseaseZHEmbedding_HNSW64.index"))
        bios_term2cid = json.load(open((os.path.join(os.environ['my_datasets_dir'], "bios_v2.2_release/CoreData/TermDiseaseZH.json"))))
        bios_terms = list(bios_term2cid.keys())
        eval_tok = BertTokenizer.from_pretrained(eval_model_path)
        eval_bert = BertModel.from_pretrained(eval_model_path)
        eval_bert.eval()
        top_k = 3
        rank0_print("calculating bios scores init time: ", datetime.now()-start_time)
        
        batch_bios_scores_p = []
        batch_bios_scores_r = []
        batch_bios_scores_f1 = []
        
        def get_terms(text_list):
            terms = []
            for text in text_list:
                terms.extend(text.split("，"))
            terms = [t.strip() for t in terms if t.strip()]
            terms = list(set(terms))
            return terms
        
        all_terms = get_terms(predictions + references)
        
        def batch_get_topk_cids(terms, bsz=8):
            all_topk_cids = {}
            batch_embeddings = []
            for i in range(0, len(terms), bsz):
                inp = eval_tok(terms[i:i+bsz], return_tensors='pt', padding=True)
                input_ids, attention_mask = inp['input_ids'].to(eval_bert.device), inp['attention_mask'].to(eval_bert.device)
                with torch.no_grad():
                    batch_embeddings.append(eval_bert(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state'][:,0].cpu().numpy())
            batch_embeddings = np.concatenate(batch_embeddings, axis=0)
            D,I = bios_index.search(batch_embeddings, top_k*10)
            for i in range(I.shape[0]):
                query_term = terms[i]
                cid_distances = {}
                for j in range(I.shape[1]):
                    bios_term = bios_terms[I[i][j]]
                    cid = bios_term2cid[bios_term]
                    cid_distances[cid] = cid_distances.get(cid, []) + [D[i][j]]
                cid_distances = sorted(list(cid_distances.items()), key=lambda x: np.mean(x[1]))
                topk_cids = set([cid for cid, dis in cid_distances[:top_k]])
                all_topk_cids[query_term] = topk_cids
            return all_topk_cids
        
        all_term2cids = batch_get_topk_cids(all_terms)
        rank0_print("calculating bios scores batch_get_topk_cids time: ", datetime.now()-start_time)
        
        def get_bios_prf(pred,ref):
            pred_terms = get_terms([pred])
            ref_terms = get_terms([ref])
            
            # precision
            precision = 0
            pred_topk_cids = [all_term2cids[term] for term in pred_terms]
            ref_topk_cids = [all_term2cids[term] for term in ref_terms]
            for pred_topk_cid in pred_topk_cids:
                if len(ref_topk_cids) == 0:
                    break
                # 优先匹配最相似的项
                intersect = [len(pred_topk_cid & ref_topk_cid) for ref_topk_cid in ref_topk_cids]
                if max(intersect) == 0:
                    continue
                else:
                    precision += 1 / len(pred_topk_cids)
                    ref_topk_cids.pop(np.argmax(intersect))
            
            # recall
            recall = 0
            pred_topk_cids = [all_term2cids[term] for term in pred_terms]
            ref_topk_cids = [all_term2cids[term] for term in ref_terms]
            for ref_topk_cid in ref_topk_cids:
                if len(pred_topk_cids) == 0:
                    break
                # 优先匹配最相似的项
                intersect = [len(pred_topk_cid & ref_topk_cid) for pred_topk_cid in pred_topk_cids]
                if max(intersect) == 0:
                    continue
                else:
                    recall += 1 / len(ref_topk_cids)
                    pred_topk_cids.pop(np.argmax(intersect))

            f1_score = 2*precision*recall/(precision+recall) if precision + recall > 0 else 0
            return precision, recall, f1_score

        for pred, ref in zip(predictions, references):
            p,r,f = get_bios_prf(pred, ref)
            batch_bios_scores_p.append(p)
            batch_bios_scores_r.append(r)
            batch_bios_scores_f1.append(f)
        
        avg_bios_score_p = sum(batch_bios_scores_p) / len(batch_bios_scores_p)
        avg_bios_score_r = sum(batch_bios_scores_r) / len(batch_bios_scores_r)
        avg_bios_score_f1 = sum(batch_bios_scores_f1) / len(batch_bios_scores_f1)
        report_metrics.update({'bios_score-p':avg_bios_score_p, 'bios_score-r':avg_bios_score_r, 'bios_score-f1':avg_bios_score_f1})
        rank0_print("calculating bios scores time: ", datetime.now()-start_time)
        
        for pred,ref,rouge_score, bertscore, bios_score in zip(predictions, references, batch_rouge_scores, batch_bert_scores[2], batch_bios_scores_f1):
            rank0_print(f"ref: {ref} pred: {pred} rougeL-f1: {rouge_score['rouge-l']['f']} bert_score_f1: {bertscore} bios_score_f1: {bios_score}")
        
        all_pred_terms = [get_terms([pred]) for pred in predictions]
        all_ref_terms = [get_terms([ref]) for ref in references]
        all_pred_topk_cids = [[all_term2cids[term] for term in terms] for terms in all_pred_terms]
        all_ref_topk_cids = [[all_term2cids[term] for term in terms] for terms in all_ref_terms]
        
        # save eval result
        if local_rank == 0:
            df = pd.DataFrame(
                {
                    'pred':original_predictions,
                    'ref':original_references,
                    'pred_terms': all_pred_terms,
                    'ref_terms': all_ref_terms,
                    'pred_topk_cid':all_pred_topk_cids,
                    'ref_topk_cid':all_ref_topk_cids,
                    'rougel_p':[s['rouge-l']['p'] for s in batch_rouge_scores],
                    'rougel_r':[s['rouge-l']['r'] for s in batch_rouge_scores],
                    'rougel_f':[s['rouge-l']['f'] for s in batch_rouge_scores],
                    'bert_scores_p':batch_bert_scores[0],
                    'bert_scores_r':batch_bert_scores[1],
                    'bert_scores_f':batch_bert_scores[2],
                    'bios_scores_p':batch_bios_scores_p,
                    'bios_scores_r':batch_bios_scores_r,
                    'bios_scores_f':batch_bios_scores_f1
                }
            )
            df.to_csv(os.path.join(training_args.output_dir, f"eval_result_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"), index=False)
        
        return report_metrics
    
    if not training_args.compute_metrics:
        compute_metrics = None
    
    # Load Trainer
    trainer = DictLLMTrainer(model=model, tokenizer=tok, args=training_args, **data_modules, compute_metrics=compute_metrics)
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")) and not debug_mode:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    if not debug_mode:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
