import json
import typing
from pathlib import Path
from tokenizers import Tokenizer
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F

REMOTE_ROOT_URL= "https://rome.baulab.info"
REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/known_1000.json"

def pad_inputs(batch, pad_token_id=None):
    '''(input_ids:list[tensor], attention_mask:list[tensor], labels:list[tensor])'''
    input_ids, attention_mask = batch[0], batch[1]
    labels = batch[2] if len(batch) == 3 else None

    max_len = max([x.shape[-1] for x in input_ids])
    # align right for gpt model
    input_ids = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=pad_token_id) for x in input_ids]).squeeze()
    attention_mask = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=0) for x in attention_mask]).squeeze()
    labels = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=-100) for x in labels]).squeeze() if labels else None

    return input_ids, attention_mask, labels
class Knowns(Dataset):
    def __init__(self, data_path: str, tokenizer : Tokenizer, size=None, *args, **kwargs,):
        self.tokenizer = tokenizer

        with open(data_path, "r") as f:
            self.data = json.load(f)
        if size:
            self.data = self.data[:min(size, len(self.data))]
        
        self.input_ids, self.attention_mask, self.labels = self.make_inputs(self.data)
        
        print(f"Loaded dataset with {len(self)} elements")
    
    def make_inputs(self, data):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for d in data:
            prompt = d['prompt'].strip()
            prediction = d['prediction'].strip()
            pred_ids = self.tokenizer.encode(prediction)
            
            res = self.tokenizer(prompt)
            input_ids, attention_mask = res['input_ids'], res['attention_mask']
            
            input_ids = torch.tensor(input_ids + [pred_ids[0]]).unsqueeze(0)
            attention_mask = torch.tensor(attention_mask + [1]).unsqueeze(0)
            labels = -100 * torch.ones_like(input_ids)
            labels[:,-1] = input_ids[:,-1]
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
        
        return input_ids_list, attention_mask_list, labels_list
    
    def collate_fn(self, batch):
        input_ids = [b[0] for b in batch]
        attention_mask = [b[1] for b in batch]
        labels = [b[2] for b in batch]
        return pad_inputs((input_ids, attention_mask, labels), self.tokenizer.pad_token_id)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]