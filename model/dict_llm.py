from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BertModel
from .llm import LLM
from .llm_hooker import LLMHooker, LLMHookerConfig
import os
import json



def Sinkhorn(self, K, u, v, thresh=1e-2):
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    for i in range(self.max_iter):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break

    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

    return T

class DictsEncoder(nn.Module):
    def __init__(self, hidden_size, output_dim, num_encoder_head, num_encoder_layers, special_tokens_path=None):
        super(DictsEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.nhead = num_encoder_head
        self.num_encoder_layers = num_encoder_layers
        self.tok = AutoTokenizer.from_pretrained(os.path.join(os.environ['my_models_dir'], 'bert-base-chinese'))
        if special_tokens_path is not None:
            special_tokens = json.load(open(special_tokens_path))
            self.tok.add_tokens(special_tokens)
        self.sep_id = self.tok.convert_tokens_to_ids('[SEP]')
        self.cls_id = self.tok.convert_tokens_to_ids('[CLS]')
        self.embedding = nn.Embedding(len(self.tok), hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, 
                nhead=num_encoder_head, 
                batch_first=True,
                dim_feedforward=hidden_size*3,
                ),
            num_layers=num_encoder_layers, 
            norm=nn.LayerNorm(hidden_size)
            )
        self.linear = nn.Linear(hidden_size, output_dim)
        self.norm = nn.LayerNorm(output_dim)
    
    def prepare_input_ids_and_mask(self, dicts : List[Dict], no_mask=False):
        spans = []
        input_ids = []
        for d in dicts:
            span = []
            # header_ids = self.tok(d['header'], add_special_tokens=False)['input_ids']
            # span.append((len(input_ids),len(input_ids)+len(header_ids)))
            # input_ids.extend(header_ids)
            for key,value in d['data'].items():
                key_ids = self.tok(key, add_special_tokens=False)['input_ids']
                value_ids = self.tok(value, add_special_tokens=False)['input_ids']
                kv_ids = key_ids + value_ids
                span.append((len(input_ids), len(input_ids)+len(kv_ids)))
                input_ids.extend(kv_ids)
            span.append(len(input_ids)) # sep_idx
            input_ids.append(self.sep_id)
            spans.append(span)
        input_ids.append(self.cls_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        mask = torch.ones((input_ids.shape[-1], input_ids.shape[-1]),dtype=torch.bool)
        if not no_mask:
            for span in spans:
                sep_idx = span[-1]
                for i,j in span[:-1]:
                    mask[i:j,i:j] = False
                    mask[i:j,sep_idx] = False
                    mask[sep_idx,i:j] = False
                mask[-1, sep_idx] = False
                mask[sep_idx, -1] = False
        else:
            mask[:,:] = False
        input_ids = input_ids.to(self.embedding.weight.data.device)
        mask = mask.to(self.embedding.weight.data.device)
        return input_ids, mask
    
    def forward(self, dicts : List[Dict], no_mask=False):
        input_ids, mask = self.prepare_input_ids_and_mask(dicts, no_mask=no_mask)
        x = self.embedding(input_ids).reshape(input_ids.shape[0], self.hidden_size)
        x = self.transformer_encoder(src=x, mask=mask)
        x = x[-1,:]
        x = self.linear(x)
        x = self.norm(x)
        return x

class DictLLM(nn.Module):
    def __init__(self, mt_path, 
                 encoder_hidden_size, 
                 num_table_token, 
                 num_encoder_head, 
                 num_encoder_layers, 
                 max_length=2048, 
                 special_tokens_path=None,
                 no_mask=False
                 ):
        super(DictLLM, self).__init__()
        self.no_mask = no_mask
        self.num_table_token = num_table_token
        self.encoder_hidden_size = encoder_hidden_size
        self.max_length = max_length
        self.llm = LLM.from_pretrained(mt_path=mt_path).float()
        self.embedding_dim = self.llm.embedding.embedding_dim
        self.dicts_encoder = DictsEncoder(encoder_hidden_size, self.embedding_dim*num_table_token, num_encoder_head, num_encoder_layers, special_tokens_path)

    def forward(self, input_text : List[str], dicts : List[List[Dict[str, str]]] = None, label_text : List[str] = None, **kwargs):
        batch_size = len(input_text)
        inp = self.llm.tok(input_text, padding=True, return_tensors='pt')
        input_ids, attention_mask, labels = inp['input_ids'], inp['attention_mask'], None
        
        if label_text is not None:
            label_text = [t + f" {self.llm.tok.eos_token}" for t in label_text]
            input_lens = attention_mask.sum(dim=1)
            all_text = [t1 + t2 for t1, t2 in zip(input_text, label_text)]
            inp = self.llm.tok(all_text, padding=True, return_tensors='pt')
            input_ids, attention_mask = inp['input_ids'], inp['attention_mask']
            all_lens = attention_mask.sum(dim=1)
            labels = torch.ones_like(input_ids) * -100
            for i in range(batch_size):
                labels[i,input_lens[i]:all_lens[i]] = input_ids[i,input_lens[i]:all_lens[i]]

        
        input_ids = input_ids[:, :self.max_length].to(self.llm.model.device)
        inputs_embeds = self.llm.embedding(input_ids)
        
        if dicts:
            dicts_embedding = torch.vstack([self.dicts_encoder(ds, no_mask=self.no_mask) for ds in dicts]).reshape((batch_size, self.num_table_token, self.embedding_dim))
            inputs_embeds = torch.cat([dicts_embedding, inputs_embeds], dim=1)
            attention_mask = torch.cat([torch.ones((batch_size, self.num_table_token),dtype=torch.long), attention_mask], dim=1)
            if labels is not None:
                labels = torch.cat([torch.ones((batch_size, self.num_table_token),dtype=torch.long) * -100, labels],dim=1).type(torch.long)
             
        inputs_embeds = inputs_embeds[:, :self.max_length, :]
        attention_mask = attention_mask[:, :self.max_length].to(self.llm.model.device)
        if labels is not None:
            labels = labels[:, :self.max_length]
        
        model_output = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)
        return model_output

    def generate(self, input_text: List[str], dicts:List[List[Dict[str, str]]] = None, label_text: List[str] = None, cut_input=False, **genkwargs):
        input_text = input_text[0]  # only support batch_size=1 now
        inp = self.llm.tok(input_text, padding=True, return_tensors='pt')
        input_ids, attention_mask = inp['input_ids'], inp['attention_mask']
        if not dicts:
            output_ids = self.llm.model.generate(input_ids=input_ids.to(self.llm.model.device), attention_mask=attention_mask.to(self.llm.model.device), **genkwargs)
        else:
            dicts = dicts[0]
            with torch.no_grad():
                dicts_embedding = self.dicts_encoder(dicts, no_mask=self.no_mask).reshape((1, self.num_table_token, self.embedding_dim))
            input_ids = F.pad(input_ids, (self.num_table_token,0), value=self.llm.tok.pad_token_id)
            attention_mask = F.pad(attention_mask, (self.num_table_token,0), value=1)
            
            def edit_func(module, input_args, input_kwargs, output):
                if input_args[0].numel() > 1: # if is first forward, not in the process of generation
                    output[0,:self.num_table_token,:] = dicts_embedding.to(module.weight.data.device)
                return output

            with LLMHooker(self.llm, LLMHookerConfig("embedding",save_output=False, edit_output=edit_func)):
                output_ids = self.llm.model.generate(input_ids=input_ids.to(self.llm.model.device), attention_mask=attention_mask.to(self.llm.model.device), **genkwargs)
        
        if cut_input:
            output_ids = output_ids[:, input_ids.shape[-1]:]
        
        return output_ids