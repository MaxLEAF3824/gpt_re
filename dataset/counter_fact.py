import ujson as json
import jsonlines
import typing
from pathlib import Path
from tokenizers import Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

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


class CounterFact(Dataset):
    def __init__(self, data_path: str, tokenizer: Tokenizer, size=None, max_len=1024, *args, **kwargs,):
        self.tokenizer = tokenizer
        self.max_len = max_len
        if data_path.endswith('.jsonl'):
            self.data = list(jsonlines.open(data_path))
        elif data_path.endswith('.json'):
            self.data = json.load(open(data_path))
        self.data = self.data[:size] if size else self.data
        self.make_inputs(self.data)
        print(f"Loaded dataset with {len(self)} elements")

    def make_inputs(self, data):
        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        self.prompts = []
        self.target_trues = []

        for d in tqdm(data):
            rr = d['requested_rewrite']
            prompt = rr['prompt'].strip()
            subject = rr['subject'].strip()
            target_true = rr['target_true']['str'].strip()
            target = rr['target_new']['str'].strip()
            # target = target_true
            prompt = prompt.replace('{}', subject)
            input_text = prompt + ' ' + target

            res = self.tokenizer(input_text, return_tensors='pt')
            input_ids, attention_mask = res['input_ids'], res['attention_mask']
            
            if input_ids.shape[1] > self.max_len:
                continue

            labels = -100 * torch.ones_like(input_ids)
            target_ids = self.tokenizer(target, return_tensors='pt')['input_ids']
            labels[:,-target_ids.shape[-1]:] = input_ids[:,-target_ids.shape[-1]:]
            
            self.input_ids.append(input_ids)
            self.attention_mask.append(attention_mask)
            self.target_trues.append(target_true)
            self.labels.append(labels)
            self.prompts.append(prompt)
            

    def collate_fn(self, batch):
        input_ids = [b[0] for b in batch]
        attention_mask = [b[1] for b in batch]
        labels = [b[2] for b in batch]
        return pad_inputs((input_ids, attention_mask, labels), self.tokenizer.pad_token_id)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]