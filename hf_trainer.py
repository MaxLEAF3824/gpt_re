import os
from dataclasses import dataclass, field, asdict
import json
import pathlib
from typing import Dict, Optional, Sequence, Tuple, Union, List, Any
import torch.nn.functional as F
import torch
import transformers
from transformers import Trainer, TrainingArguments
from torch.distributed.elastic.multiprocessing.errors import record
import torch.nn as nn
import numpy as np
import random
import torch
from model.llm_utils import LoadWoInit
from model.dict_llm import DictLLM
from rouge_chinese import Rouge
import jieba
import time
import bert_score


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
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def load_hf_model_tok(mt_path):
    tok = transformers.AutoTokenizer.from_pretrained(mt_path, trust_remote_code=True)
    with LoadWoInit():
        model = transformers.AutoModelForCausalLM.from_pretrained(mt_path, trust_remote_code=True)
    return model, tok


@dataclass
class ModelArguments:
    mt_path: str = field(default=None)
    encoder_hidden_size : int = field(default=768)
    num_table_token : int = field(default=5)
    num_encoder_head : int = field(default=8)
    num_encoder_layers : int = field(default=12)
    max_length : int = field(default=1600)
    special_tokens_path : str = field(default=None)
    mask_strategy : str = field(default="hierarchical")
    position_strategy : str = field(default="group")
    encoder_type : str = field(default="transformer")
    mapper_type : str = field(default="linear")

@dataclass
class DataArguments:
    train_data_path : str = field(default=None)
    eval_data_path : Optional[str] = field(default=None)


@dataclass
class TrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch")
    remove_unused_columns : bool = False
    output_dir : str = field(default="output/")
    
@dataclass
class DictDataCollator(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, object]:
        input_text = []
        dicts = []
        label_text = []
        for i in instances:
            input_text.append(i['input'])
            if 'data' in i:
                dicts.append(i['data'])
            if 'output' in i:
                label_text.append(i['output'])
        return dict(input_text=input_text, dicts=dicts, label_text=label_text)


class DictLLMTrainer(Trainer):
    def compute_loss(self, dllm, inputs, return_outputs=False, **kwargs): 
        outputs = dllm(**inputs, **kwargs)
        return (outputs['loss'], outputs) if return_outputs else outputs['loss']
    
    def prediction_step(
        self,
        dllm: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        labels = dllm.llm.tok(inputs["label_text"], padding=True, return_tensors='pt', add_special_tokens=False)['input_ids'].to(dllm.llm.model.device)
        
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(dllm, inputs, return_outputs=True, output_attentions=True)
                loss = loss.mean().detach()
                output_ids = dllm.generate(**inputs, cut_input=True, max_new_tokens=2*labels.shape[-1], synced_gpus=True).to(dllm.llm.model.device)
        
        return (loss, output_ids, labels)


@record
def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    rank0_print(f"model_args: {model_args}\ndata_args: {data_args}")
    
    # Load Dataset
    train_dst = json.load(open(data_args.train_data_path))
    rank0_print('train_dst size: ', len(train_dst))
    
    eval_dst = json.load(open(data_args.eval_data_path))
    rank0_print('eval_dst size: ', len(eval_dst))
    
    data_collator = DictDataCollator()
    data_modules = dict(train_dataset=train_dst, eval_dataset=eval_dst, data_collator=data_collator)
    
    # Load Model
    dllm = DictLLM(**asdict(model_args))
    model, tok = dllm, dllm.llm.tok
    
    # compute_metrics
    def compute_metrics(EvalPrediction):
        output_ids = EvalPrediction.predictions
        output_ids = output_ids.tolist()
        output_ids = [[i for i in ids if i != -100] for ids in output_ids]
        predictions = tok.batch_decode(output_ids, skip_special_tokens=True)
        predictions = [' '.join(jieba.cut(p)) for p in predictions]
        predictions = ["--" if not p.strip() else p for p in predictions]
        
        label_ids = EvalPrediction.label_ids
        label_ids = label_ids.tolist()
        label_ids = [[i for i in ids if i != -100] for ids in label_ids]
        references = tok.batch_decode(label_ids, skip_special_tokens=True)
        references = [' '.join(jieba.cut(r)) for r in references]
        
        rouge = Rouge()
        rouge_scores = rouge.get_scores(predictions, references)
        rougel_f1 = sum([s['rouge-l']['f'] for s in rouge_scores])/len(rouge_scores)
        rougel_p = sum([s['rouge-l']['p'] for s in rouge_scores])/len(rouge_scores)
        rougel_r = sum([s['rouge-l']['r'] for s in rouge_scores])/len(rouge_scores)
        
        bert_scores = [s.cpu().tolist() for s in bert_score.score(predictions, references, lang='zh')] # p,r,f
        bert_score_p = sum(bert_scores[0])/len(bert_scores[0])
        bert_score_r = sum(bert_scores[1])/len(bert_scores[1])
        bert_score_f1 = sum(bert_scores[2])/len(bert_scores[2])
        
        for pred,ref,rouge_score,bscore in zip(predictions, references, rouge_scores, bert_scores[2]):
            rank0_print(f"ref: {ref} pred: {pred} rougeL-f1: {rouge_score['rouge-l']['f']} bert_score_f1: {bscore}")
        
        return {'rougeL' : rougel_f1, 'rougeL-p' : rougel_p, 'rougeL-r':rougel_r, 'bert_score-p':bert_score_p, 'bert_score-r':bert_score_r, 'bert_score-f1':bert_score_f1}
    
    # Load Trainer
    trainer = DictLLMTrainer(model=model, tokenizer=tok, args=training_args, **data_modules, compute_metrics=compute_metrics)
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
