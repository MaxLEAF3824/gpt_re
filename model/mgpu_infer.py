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
    
    output_texts = []
    
    for d in tqdm(local_dst, total=len(local_dst)):
        input_ids = tok(d['input'], return_tensors='pt')['input_ids']
        input_len = input_ids.shape[-1]
        input_ids = input_ids.to(model.device)
        output = model.generate(input_ids=input_ids, max_new_tokens=mnt, do_sample=False)
        output_text = tok.decode(output[0,input_len:])
        d['output'] = output_text
        output_texts.append(d)

    json.dump(output_texts, open(f'{save_path}_{local_rank}.json', 'w'), indent=4)
    # torch.distributed.barrier()
    
    rank0_print("rank 0 collecting results...")
    if local_rank == 0:
        done = False
        while not done:
            done = sum([osp.exists(f'{save_path}_{i}.json') for i in range(world_size)]) == world_size
            time.sleep(0.5)
        output_texts = []
        for i in range(world_size):
            output_texts += json.load(open(f'{save_path}_{i}.json', 'r'))
            os.remove(f'{save_path}_{i}.json')
        json.dump(output_texts, open(save_path, 'w'), indent=4)

def multigpu_inference(model, tok, dst, save_path, local_bsz=4, **kwargs):
    local_rank = get_local_rank()
    world_size = get_world_size()
    local_start = len(dst) // world_size*local_rank
    local_end = len(dst) // world_size*(local_rank+1) if local_rank != world_size-1 else len(dst)
    local_dst = dst[local_start:local_end]
    
    if local_rank == 0:
        for i in range(world_size):
            if osp.exists(f'{save_path}_{i}.json'):
                os.remove(f'{save_path}_{i}.json')
    
    output_loss = []
    
    batch_local_dst = [local_dst[i:i+local_bsz] for i in range(0, len(local_dst), local_bsz)]
    
    for batch_d in tqdm(batch_local_dst, total=len(batch_local_dst)):
        input_lens = [len(tok(d['input'])['input_ids'] for d in batch_d)]
        all_lens = [len(tok(d['input']+d['gt'])['input_ids'] for d in batch_d)]
        input_ids = tok([d['input']+d['gt'] for d in batch_d], return_tensors='pt', padding=True)['input_ids']
        output_logits = model(input_ids=input_ids.to(model.device))['logits']
        for i in range(len(batch_d)):
            start = input_lens[i]
            end = all_lens[i]
            d = batch_d[i]
            label = input_ids[i, start:end+1]
            pred = output_logits[i, start-1:end]
            loss = torch.nn.functional.cross_entropy(pred, label)
            d['loss'] = loss.item()
            output_loss.append(d)
    
    json.dump(output_texts, open(f'{save_path}_{local_rank}.json', 'w'), indent=4)
    # torch.distributed.barrier()
    
    rank0_print("rank 0 collecting results...")
    if local_rank == 0:
        done = False
        while not done:
            done = sum([osp.exists(f'{save_path}_{i}.json') for i in range(world_size)]) == world_size
            time.sleep(0.5)
        output_texts = []
        for i in range(world_size):
            output_texts += json.load(open(f'{save_path}_{i}.json', 'r'))
            os.remove(f'{save_path}_{i}.json')
        json.dump(output_texts, open(save_path, 'w'), indent=4)


def mgpu_infer(model_path, dst_path, func="generate", save_path="result.json", seed=42, **kwargs):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.distributed.init_process_group(backend="nccl")
    world_size = get_world_size()
    local_rank = get_local_rank()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    
    rank0_print(f"model_path: {model_path}\n",
                f"save_path: {save_path}\n",
                f"bos_token:{tokenizer.bos_token_id} {tokenizer.bos_token}\n",
                f"eos_token:{tokenizer.eos_token_id} {tokenizer.eos_token}\n",
                f"world_size: {world_size}\n",
                f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'None')}\n",
                f"kwargs: {kwargs}\n")
    
    with LoadWoInit():
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda(local_rank)
    
    dst = json.load(open(dst_path))
    rank0_print('dst length: ', len(dst))
    
    if func == "generate":
        multigpu_generate(model=model, tok=tokenizer, dst=dst, save_path=save_path, **kwargs)
    elif func == 'inference':
        multigpu_inference(model=model, tok=tokenizer, dst=dst, save_path=save_path, **kwargs)
    
    rank0_print(f"all done, result saved to: {save_path}")

if __name__ == '__main__':
    fire.Fire(mgpu_infer)
