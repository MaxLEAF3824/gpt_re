from .model_interface import LLM
import torch
import os
from copy import deepcopy
from pyecharts import options as opts
from pyecharts.charts import Bar, Timeline, Tab, Page
from pyecharts.faker import Faker
import ipywidgets as widgets
from IPython.display import display
import torch.nn as nn

SAVED_MODULES = ['layer', 'attn', 'mlp']

class Unembedding(nn.Module):
    def __init__(self, lm_head, ln_f):
        super().__init__()
        self.lm_head = lm_head
        self.lm_head.requires_grad_(False)
        self.ln_f = ln_f
        self.ln_f.requires_grad_(False)
        
    def forward(self, x):
        return self.lm_head(self.ln_f(x))
        
class FlowVisualizer:
    def __init__(self, mt: LLM, max_unembed_num=1000):
        self.mun = max_unembed_num
        self.mt = mt
        self.idx2token = [f"{i}-{self.mt.tokenizer.decode(i)}" for i in range(self.mt.tokenizer.vocab_size)]
        self.unembedding = Unembedding(deepcopy(mt.lm_head).to('cpu').float(), deepcopy(mt.ln_f).to('cpu').float())
        self.init_save_hook()
        self.sentences = []
        self.next_tokens = []
        self.inp_len = []
        self.layer_matrixs = []  # [batch, n_layer, seq_len, hidden_size]
        self.attn_matrixs = []
        self.mlp_matrixs = []
        self.layer_utokens = [] # unembedding tokens string for [batch, seq_len, max_unembed_num]
        self.layer_uprobs = [] # [batch, seq_len, n_layer, max_unembed_num]
        self.attn_utokens = []
        self.attn_uprobs = []
        self.mlp_utokens = []
        self.mlp_uprobs = []
        
        
    def init_save_hook(self):
        self.mt.clear_hook()
        hook_config = {
            "retain_output": True,
            "retain_input": False,
            "edit_output": None,
            "clone": True,
            "float": True,
            "detach": True,
            "device": "cpu"
        }
        for l in range(self.mt.n_layer):
            for h in SAVED_MODULES:
                self.mt.add_hook(module=getattr(self.mt, h+'s')[l], name=f'{h}_{l}', **hook_config)

    def generate(self, input_texts, **gen_wargs):
        input_texts = input_texts if isinstance(input_texts, list) else [input_texts]
        inps = [self.mt.tokenizer(text, return_tensors='pt') for text in input_texts]
        
        for inp in inps:
            input_ids, attention_mask = inp['input_ids'], inp['attention_mask']
            self.inp_len.append(input_ids.shape[1])

            # generate
            hook_idxs = [len(h.outputs) for h in self.mt.hooks.values()]
            with torch.no_grad():
                input_ids = input_ids.to(self.mt.model.device)
                attention_mask = attention_mask.to(self.mt.model.device)
                gen_wargs['max_new_tokens'] = 10 if 'max_new_tokens' not in gen_wargs else gen_wargs['max_new_tokens']
                output_ids = self.mt.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_wargs)
            
            # merge hook outputs
            for (hook, idx) in zip(self.mt.hooks.values(), hook_idxs):
                hook.outputs[idx] = torch.cat([o for o in hook.outputs[idx:]], dim=1)
                hook.outputs = hook.outputs[:idx+1]
            
            # save matrix and utokens
            for h in SAVED_MODULES:
                matrixs = getattr(self, h+'_matrixs')
                module_utokens = getattr(self, h+'_utokens')
                module_uprobs = getattr(self, h+'_uprobs')
                
                # save matrix
                cur_matrix = torch.cat([self.mt.hooks[f'{h}_{l}'].outputs[-1] for l in range(self.mt.n_layer)], dim=0)
                matrixs.append(cur_matrix)

                cur_uprob = torch.softmax(self.unembedding(cur_matrix), dim=-1)  # [n_layer, seq_len, vocab_size]
                
                # cutoff utokens (top k prob diff)
                cur_udiff = (cur_uprob[1:] - cur_uprob[:-1]).abs().sum(dim=0) # [seq_len, vocab_size]
                cur_uids = torch.topk(cur_udiff, k=self.mun, dim=-1, sorted=True).indices # [seq_len, max_unembed_num]
                
                seq_len = cur_uids.shape[0]
                cur_utokens = []
                cur_uprobs = []
                for j in range(seq_len):
                    uids = cur_uids[j]
                    uprobs = torch.index_select(cur_uprob[:,j,:], dim=-1, index=uids) # [n_layer, max_unembed_num]
                    utokens = [f"{self.idx2token[id]}" for id in uids] # [max_unembed_num]
                    uprobs = uprobs.cpu().numpy().tolist()
                    cur_utokens.append(utokens) # [seq_len, max_unembed_num]
                    cur_uprobs.append(uprobs) # [seq_len, n_layer, max_unembed_num]
                module_utokens.append(cur_utokens) # [batch, seq_len, max_unembed_num]
                module_uprobs.append(cur_uprobs) # [batch, seq_len, n_layer, max_unembed_num]
            out_tokens = self.mt.tokenizer.batch_decode(output_ids[0])
            self.sentences.append(out_tokens[:-1])
            self.next_tokens.append(out_tokens[-1])
        self.mt.reset_hook()

    def visual_utokens(self, sidx=-1, module='layer', unum=10):
        assert module in SAVED_MODULES
        cur_sentence = self.sentences[sidx]
        tab = Tab()
        for tidx in range(len(cur_sentence)):
            module_utokens = getattr(self, module+'_utokens')
            module_uprobs = getattr(self, module+'_uprobs')
            tl = Timeline()
            for l in range(self.mt.n_layer):
                cur_utokens = module_utokens[sidx][tidx][:unum]
                cur_uprobs = module_uprobs[sidx][tidx][l][:unum]
                bar = (
                    Bar()
                    .add_xaxis(cur_utokens)
                    .add_yaxis(module, cur_uprobs, label_opts=opts.LabelOpts(position="right"))
                    .reversal_axis()
                    .set_global_opts(
                        title_opts={"text": f"Unembedding Token Flow: {module}"},
                        xaxis_opts=opts.AxisOpts(name="Probability"),
                        yaxis_opts=opts.AxisOpts(name="Top k Unembedding Tokens")
                    )
                )
                tl.add(bar, f"{l}")
            tab.add(tl, cur_sentence[tidx])
        return tab
    
    def get_similar_token(self, token_id, k=20):
        embedding = self.mt.embedding.weight.data
        cos_values, cos_indices = torch.topk(torch.cosine_similarity(embedding, embedding[token_id].unsqueeze(0), dim=1),k=k)
        return [f"{i}.{self.idx2token[id]}: {cos_values[i].item():.3f}" for i, id in enumerate(cos_indices)]
        
    def clear(self):
        self.sentences.clear()
        self.next_tokens.clear()
        self.inp_len.clear()
        self.layer_matrixs.clear()  
        self.attn_matrixs.clear()
        self.mlp_matrixs.clear()
        self.layer_utokens.clear() 
        self.layer_uprobs.clear() 
        self.attn_utokens.clear()
        self.attn_uprobs.clear()
        self.mlp_utokens.clear()
        self.mlp_uprobs.clear()