from typing import Dict, List
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from .llm import LLM
from .llm_hooker import LLMHooker, LLMHookerConfig
import os

SPECIAL_TOKENS=['[正常]','[异常]','[敏感]','[耐药]','[中介]','[极弱阳性]','[弱阳性]','[阴性]','[阳性]']

class DictsEncoder(nn.Module):
    def __init__(self, hidden_size, output_dim, num_encoder_head, num_encoder_layers):
        super(DictsEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.nhead = num_encoder_head
        self.num_encoder_layers = num_encoder_layers
        self.tok = AutoTokenizer.from_pretrained(os.path.join(os.environ['my_models_dir'], 'bert-base-chinese'))
        self.tok.add_tokens(SPECIAL_TOKENS)
        self.sep_id = self.tok.convert_tokens_to_ids('[SEP]')
        self.cls_id = self.tok.convert_tokens_to_ids('[CLS]')
        self.embedding = nn.Embedding(len(self.tok), hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_encoder_head), num_layers=num_encoder_layers)
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
        input_ids = input_ids.to(self.embedding.weight.data.device)
        mask = mask.to(self.embedding.weight.data.device)
        return input_ids, mask
    
    def forward(self, dicts : List[Dict]):
        input_ids, mask = self.prepare_input_ids_and_mask(dicts)
        x = self.embedding(input_ids).reshape(input_ids.shape[0], self.hidden_size)
        x = self.transformer_encoder(src=x, mask=mask)
        x = x[-1,:]
        x = self.linear(x)
        return x

class DictLLM(nn.Module):
    def __init__(self, mt_path, encoder_hidden_size, num_table_token, num_encoder_head, num_encoder_layers):
        super(DictLLM, self).__init__()
        self.num_table_token = num_table_token
        self.encoder_hidden_size = encoder_hidden_size
        self.llm = LLM.from_pretrained(model_path=mt_path)
        self.embedding_dim = self.llm.embedding.embedding_dim
        self.table_token_id = self.llm.tok.unk_token_id
        self.dicts_encoder = DictsEncoder(encoder_hidden_size, self.embedding_dim*num_table_token, num_encoder_head, num_encoder_layers)
        
    def forward(self, input_text : List[str], dicts : List[List[Dict]], label_text : List[str] = None):
        batch_size = len(input_text)
        inp = self.llm.tok(input_text, padding=True, return_tensors='pt')
        input_ids, attention_mask = inp['input_ids'], inp['attention_mask']
        labels = None
        if label_text:
            input_lens = attention_mask.sum(dim=1)
            all_text = [t1+t2 for t1,t2 in zip(input_text, label_text)]
            inp = self.llm.tok(all_text, padding=True, return_tensors='pt')
            input_ids, attention_mask = inp['input_ids'], inp['attention_mask']
            all_lens = attention_mask.sum(dim=1)
            labels = torch.ones_like(input_ids) * -100
            for i in range(batch_size):
                labels[i,input_lens[i]:all_lens[i]] = input_ids[i,input_lens[i]:all_lens[i]]
            print('labels: ', labels.shape)
        input_ids = input_ids.to(self.llm.model.device)
        
        text_embedding = self.llm.embedding(input_ids)
        dicts_embedding = torch.vstack([self.dicts_encoder(ds) for ds in dicts]).reshape((batch_size, self.num_table_token, self.embedding_dim))
        
        input_embeds = torch.cat([text_embedding, dicts_embedding], dim=1)
        print('input_embeds: ', input_embeds.shape)
        
        attention_mask = torch.cat([torch.ones((batch_size, self.num_table_token)), attention_mask], dim=1).to(self.llm.model.device)
        print('attention_mask: ', attention_mask.shape)
        
        if labels is not None:
            labels = torch.cat([torch.ones((batch_size, self.num_table_token)) * -100, labels],dim=1).to(self.llm.model.device)
            labels = labels.type(torch.long)
            print('labels: ', labels.shape)
        
        model_output = self.llm(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
        return model_output

    def generate(self, input_text: str, dicts:List[Dict], **genkwargs):
        with torch.no_grad():
            dicts_embedding = self.dicts_encoder(dicts).reshape((1, self.num_table_token, self.embedding_dim))
        input_ids = self.llm.tok(input_text)['input_ids']
        table_token_ids = [self.table_token_id for i in range(self.num_table_token)]
        input_ids = table_token_ids + input_ids
        input_ids = torch.tensor(input_ids).reshape(1,-1)
        attention_mask = torch.ones_like(input_ids)
        
        def edit_func(module, input, output):
            if input[0].numel() > 1:
                input_ids = list(input[0].squeeze().numpy())
                print(f"edit embedding is called: original_input:{input_ids}, original_output:{output.shape}")
                nt = self.num_table_token
                table_start_idxs = [i for i in range(len(input_ids)-nt) if input_ids[i:i+nt] == table_token_ids]
                for idx in table_start_idxs:
                    output[0,idx:idx+nt,:] = dicts_embedding
            return output

        with LLMHooker(self.llm, LLMHookerConfig("embedding",retain_output=False, edit_output=edit_func)):
            output = self.llm.model.generate(input_ids=input_ids, attention_mask=attention_mask, **genkwargs)
        return output


if __name__=="__main__":
    import json
    dst = json.load(open("/mnt/petrelfs/guoyiqiu/coding/my_datasets/ninth/checkout_data_sample.json"))
    print('dst: ', len(dst))
    dllm = DictLLM(mt_path="/mnt/petrelfs/guoyiqiu/coding/my_models/gpt2", encoder_hidden_size=128,num_table_token=5)
    dllm.float()
    inputs = []
    for d in dst:
        text = "Hello! How can I help you?"
        dicts = d['完整化验结果']
        inputs.append({"input_text":text, "dicts":dicts})
    forward_output = dllm(**inputs[0])
    print('forward_output: ', forward_output)
    generate_output = dllm.generate(**inputs[0])
    print('generate_output: ', generate_output)
    