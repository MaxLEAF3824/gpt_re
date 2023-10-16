import os
import os.path as osp
import fire
from pprint import pprint
import torch
import json
import jsonlines
import random
from transformers import GenerationConfig,AutoTokenizer,AutoModelForCausalLM
import pickle
import csv
from tqdm import tqdm
import time
from datasets import load_dataset
from llm_utils import *


def get_local_rank():
    return int(os.getenv('LOCAL_RANK', '0'))

def get_rank():
    return int(os.getenv('RANK', '0'))

def get_world_size():
    return int(os.getenv('WORLD_SIZE', '1'))

def rank0_print(*args):
    if get_local_rank() == 0:
        print(*args)

def multigpu_generate(model, tok, dst, save_path, mnt, **kwargs):
    local_rank = get_local_rank()
    world_size = get_world_size()
    local_start = len(dst) // world_size*local_rank
    local_end = len(dst) // world_size*(local_rank+1) if local_rank != world_size-1 else len(dst)
    local_dst = dst[local_start:local_end]
    
    if local_rank == 0:
        for i in range(world_size):
            if osp.exists(f'{save_path}_{i}.json'):
                os.remove(f'{save_path}_{i}.json')
    
    outputs = []
    
    for d in tqdm(local_dst, total=len(local_dst)):
        input_ids = tok(d['input'], return_tensors='pt')['input_ids']
        input_len = input_ids.shape[-1]
        input_ids = input_ids.to(model.device)
        output = model.generate(input_ids=input_ids, max_new_tokens=mnt, do_sample=False)
        output_text = tok.decode(output[0,input_len:])
        d['output'] = output_text
        outputs.append(d)

    json.dump(outputs, open(f'{save_path}_{local_rank}.json', 'w'), indent=4)


def multigpu_inference(model, tok, dst, save_path, local_bsz=6,**kwargs):
    local_rank = get_local_rank()
    world_size = get_world_size()
    local_start = len(dst) // world_size*local_rank
    local_end = len(dst) // world_size*(local_rank+1) if local_rank != world_size-1 else len(dst)
    local_dst = dst[local_start:local_end]
    
    if local_rank == 0:
        for i in range(world_size):
            if osp.exists(f'{save_path}_{i}.json'):
                os.remove(f'{save_path}_{i}.json')
    
    outputs = []
    
    for d in tqdm(local_dst, total=len(local_dst)):
        input_ids = tok(d['input'], return_tensors='pt')['input_ids']
        prompt_ids = input_ids[...,: -1]
        last_token_id = input_ids[...,-1]
        labels = d['labels'] if isinstance(d['labels'], list) else [d['labels']]
        batch_labels_list = [labels[i:i+local_bsz] for i in range(0, len(labels), local_bsz)]
        past_key_values = model(input_ids=prompt_ids.to(model.device))['past_key_values']
        
        loss_list = []
        
        for batch_labels in batch_labels_list:
            batch_label_length = [len(tok(l, add_special_tokens=False)['input_ids']) for l in batch_labels]
            batch_label_ids = tok(batch_labels, return_tensors='pt', padding=True, add_special_tokens=False)['input_ids']
            batch_input_ids = torch.cat((last_token_id.repeat(len(batch_labels),1), batch_label_ids), dim=-1)
            batch_past_key_values = [(k.repeat(len(batch_labels),1,1,1), v.repeat(len(batch_labels),1,1,1)) for (k,v) in past_key_values]
            batch_logits = model(input_ids=batch_input_ids.to(model.device), past_key_values=batch_past_key_values)['logits']
            batch_label_ids = batch_label_ids.to(model.device)
            
            for i in range(len(batch_labels)):
                length = batch_label_length[i]
                label = batch_label_ids[i, :length]
                pred = batch_logits[i, :length]
                loss = torch.nn.functional.cross_entropy(pred, label)
                loss_list.append(loss.item())
        
        d['label_loss'] = loss_list
        outputs.append(d)
    
    json.dump(outputs, open(f'{save_path}_{local_rank}.json', 'w'), indent=4)


def mgpu_infer(model_path, dst_path, save_path="result.json", func="gen", seed=42, **kwargs):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.distributed.init_process_group(backend="nccl")
    world_size = get_world_size()
    local_rank = get_local_rank()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if func == "gen":
        func = multigpu_generate
    elif func == 'infer':
        func = multigpu_inference
    else:
        raise ValueError(f"func {func} not supported.")
    
    rank0_print(f"model_path: {model_path}\n",
                f"dst_path: {dst_path}\n",
                f"save_path: {save_path}\n",
                f"func: {func.__name__}\n",
                f"seed: {seed}\n",
                f"kwargs: {kwargs}\n",
                f"bos_token:{tokenizer.bos_token_id} {tokenizer.bos_token}\n",
                f"eos_token:{tokenizer.eos_token_id} {tokenizer.eos_token}\n",
                f"pad_token:{tokenizer.pad_token_id} {tokenizer.pad_token}\n",
                f"world_size: {world_size}\n",
                f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'None')}\n"
                )
    
    with LoadWoInit():
        model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                     torch_dtype=torch.float16, 
                                                     trust_remote_code=True,
                                                     use_f).cuda(local_rank)
    
    dst = json.load(open(dst_path))
    rank0_print('dst size: ', len(dst))
    
    func(model=model, tok=tokenizer, dst=dst, save_path=save_path, **kwargs)
    
    rank0_print("rank 0 waiting for collecting results...")
    
    if local_rank == 0:
        done = False
        while not done:
            done = sum([osp.exists(f'{save_path}_{i}.json') for i in range(world_size)]) == world_size
            time.sleep(0.5)
        outputs = []
        for i in range(world_size):
            outputs += json.load(open(f'{save_path}_{i}.json', 'r'))
            os.remove(f'{save_path}_{i}.json')
        json.dump(outputs, open(save_path, 'w'), indent=4)
    
    rank0_print(f"all done.")

if __name__ == '__main__':
    fire.Fire(mgpu_infer)
