import os
from dataclasses import dataclass, field
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

torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)
np.random.seed(42)

flag = True

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def load_hf_model_tok(mt_path):
    tok = transformers.Autotok.from_pretrained(mt_path, trust_remote_code=True)
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
    def compute_loss(self, dllm, inputs, return_outputs=False): 
        outputs = dllm(**inputs, output_attentions=True)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        input_text = inputs["input_text"]
        print('input_text: ', input_text)
        label_text = inputs["label_text"]
        print('label_text: ', label_text)
        labels = model.llm.tok(label_text, padding=True, return_tensors='pt', add_special_tokens=False)['input_ids'].to(model.llm.model.device)
        print('labels: ', labels)
        
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
            output_text = model.generate(**inputs, cut_input=True, max_new_tokens=2*labels.shape[-1])
            print('output_text: ', output_text)
            output_ids = model.llm.tok(output_text, padding=True, return_tensors='pt', add_special_tokens=False)['input_ids'].to(model.llm.model.device)
            print('output_ids: ', output_ids)
        
        # return (loss, outputs['logits'], labels)
        return (loss, output_ids, labels)

    def compute_metrics(self, EvalPrediction):
        print(EvalPrediction.predictions)
        print(EvalPrediction.label_ids)
        assert False

    

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
    dllm = DictLLM(model_args.mt_path, model_args.encoder_hidden_size, model_args.num_table_token, model_args.num_encoder_head, model_args.num_encoder_layers, max_length=model_args.max_length)
    model, tok = dllm, dllm.llm.tok
    
    # Load Trainer
    trainer = DictLLMTrainer(model=model, tokenizer=tok, args=training_args, **data_modules)
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()
    
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
