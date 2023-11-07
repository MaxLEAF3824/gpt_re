from typing import Dict, List
from model.llm import LLM
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import torch.nn.functional as F

class DictsEncoder(nn.Module):
    def __init__(self, hidden_size, output_dim, nhead=8, num_layers=6):
        super(DictsEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.tok = AutoTokenizer.from_pretrained('/mnt/petrelfs/guoyiqiu/coding/huggingface/hub/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55')
        self.tok.add_tokens(['[严重偏低]','[偏低]','[正常]','[偏高]','[严重偏高]','[异常]'])
        self.sep_id = self.tok.convert_tokens_to_ids('[SEP]')
        self.cls_id = self.tok.convert_tokens_to_ids('[CLS]')
        self.embedding = nn.Embedding(len(self.tok), hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead), num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, output_dim)
        
    def prepare_input_ids_and_mask(self, dicts : List[Dict]):
        spans = []
        input_ids = []
        for d in dicts:
            span = []
            # header_ids = self.tok(d['header'], add_special_tokens=False)['input_ids']
            # span.append((len(input_ids),len(input_ids)+len(header_ids)))
            # input_ids.extend(header_ids)
            for key,value in d['values'].items():
                key_ids = self.tok(key, add_special_tokens=False)['input_ids']
                value_ids = self.tok(value, add_special_tokens=False)['input_ids']
                assert len(value_ids) == 1, "value_ids should be a single token"
                kv_ids = key_ids + value_ids
                span.append((len(input_ids), len(input_ids)+len(kv_ids)))
                input_ids.extend(kv_ids)
            span.append(len(input_ids)-1) # sep_idx
            input_ids.append(self.sep_id)
            spans.append(span)
        input_ids.append(self.cls_id)
        input_ids = torch.tensor(input_ids)
        mask = torch.ones((input_ids.shape[-1], input_ids.shape[-1]),dtype=torch.bool)
        for span in spans:
            sep_idx = span[-1]
            for i,j in span[:-1]:
                mask[i:j,i:j] = False
                mask[i:j,sep_idx] = False
                mask[sep_idx,i:j] = False
                mask[-1, sep_idx] = False
                mask[sep_idx, -1] = False
        return input_ids, mask
    
    def forward(self, dicts : List[Dict]):
        input_ids, mask = self.prepare_input_ids_and_mask(dicts)
        x = self.embedding(input_ids).squeeze()
        print('src: ', x.shape)
        x = self.transformer_encoder(src=x, mask=mask)
        x = x[-1,:]
        x = self.linear(x)
        return x
        
class DictLLM(nn.Module):
    def __init__(self, mt_path, encoder_hidden_size, num_encoding_tokens=5, **encoder_kwargs):
        super(DictLLM, self).__init__()
        self.mt = LLM.from_pretrained(model_path=mt_path)
        self.num_encoding_tokens = num_encoding_tokens
        self.encoder_hidden_size = encoder_hidden_size
        self.embedding_dim = self.mt.embedding.embedding_dim
        self.dicts_encoder = DictsEncoder(hidden_size=encoder_hidden_size, output_dim=self.embedding_dim*num_encoding_tokens, **encoder_kwargs)
        
        
    def forward(self, inputs):
        input_texts = [i[0] for i in inputs]
        dicts_list = [i[1] for i in inputs]
        batch_size = len(inputs)
        inp = self.mt.tok(input_texts, padding=True, return_tensors='pt')
        input_ids, attention_mask = inp['input_ids'], inp['attention_mask']
        print('input_ids: ', input_ids.shape)
        print('attention_mask: ', attention_mask.shape)
        text_embedding = self.mt.embedding(input_ids)
        dicts_embedding = torch.vstack([self.dicts_encoder(dicts) for dicts in dicts_list]).reshape((batch_size, self.num_encoding_tokens, self.embedding_dim))
        td_embedding = torch.cat([text_embedding, dicts_embedding], dim=1)
        td_attention_mask = torch.cat([torch.ones((batch_size, self.num_encoding_tokens)), attention_mask],dim=1)
        print('text_embedding: ', text_embedding.shape)
        print('td_embedding: ', td_embedding.shape)
        print('td_attention_mask: ', td_attention_mask.shape)
        
        
        

if __name__=="__main__":
    import json
    dst = json.load(open("/mnt/petrelfs/guoyiqiu/coding/my_datasets/ninth/checkout_data_sample.json"))
    print('dst: ', len(dst))
    dllm = DictLLM(mt_path="/mnt/petrelfs/guoyiqiu/coding/my_models/gpt2", encoder_hidden_size=128)
    inputs = []
    for d in dst:
        text = "入院时主要症状及体征:" + d['入院时主要症状及体征'] + "特殊检查及重要会诊:" + d['特殊检查及重要会诊'] 
        dicts = d['完整化验结果']
        inputs.append([text, dicts])

    dllm(inputs)
    