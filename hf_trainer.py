import os
import copy
from tqdm.auto import tqdm
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence, Tuple, Union, List, Any
import torch.nn.functional as F
import pdb
import torch
import wandb
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
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
    num_table_token : int = field(default=10)
    num_encoder_head : int = field(default=8)
    num_encoder_layers : int = field(default=12)

@dataclass
class DataArguments:
    train_data_path : str = field(default=None)
    eval_data_path : Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=2048)
    output_dir : str = field(default="output/")

class DictLLMTrainer(Trainer):
    def compute_loss(self, dllm, inputs, return_outputs=False): 
        outputs = dllm(**inputs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss

@dataclass
class DictDataCollator(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        print('instances: ', instances)
        input_text = [i['input'] for i in instances]
        dicts = [i['data'] for i in instances]
        label_text = [i['output'] for i in instances]
        return dict(input_text=input_text, dicts=dicts, label_text=label_text)


@record
def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    rank0_print(f"model_args: {model_args}\ndata_args: {data_args}")
    
    # Load Model
    dllm = DictLLM(model_args.mt_path, model_args.encoder_hidden_size, model_args.num_table_token, model_args.num_encoder_head, model_args.num_encoder_layers)
    dllm.half()
    model, tok = dllm, dllm.llm.tok
    
    # Load Dataset
    train_dst = json.load(open(data_args.train_data_path)) if data_args.train_data_path else None
    eval_dst = json.load(open(data_args.eval_data_path)) if data_args.eval_data_path else None
    data_collator = DictDataCollator()
    data_modules = dict(train_dataset=train_dst, eval_dataset=eval_dst, data_collator=data_collator)
    
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
