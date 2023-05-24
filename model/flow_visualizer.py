from .model_interface import LLM
import torch
import os
from copy import deepcopy
from pyecharts import options as opts
from pyecharts.charts import Bar, Timeline, Tab, Page
from pyecharts.faker import Faker
import ipywidgets as widgets
from IPython.display import display


SAVED_MODULES = ['layer', 'attn', 'mlp']


class FlowVisualizer:
    def __init__(self, mt: LLM, max_unembed_num=1000):
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision('medium')
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self.mun = max_unembed_num
        self.mt = mt.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.idx2token = [f"{i}-{self.mt.tokenizer.decode(i)}" for i in range(self.mt.tokenizer.vocab_size)]
        self.lm_head = deepcopy(mt.lm_head).to('cpu').float()
        self.ln_f = deepcopy(mt.ln_f).to('cpu').float()
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
            input_ids = input_ids.to(self.mt.device)
            attention_mask = attention_mask.to(self.mt.device)
        
            hook_idxs = [len(h.outputs) for h in self.mt.hooks.values()]
            
            if 'max_new_tokens' not in gen_wargs:
                gen_wargs['max_new_tokens'] = 10
            output_ids = self.mt.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_wargs)
            # out = self.mt.model(input_ids=input_ids, attention_mask=attention_mask)
        
            # merge hook outputs
            for (hook, idx) in zip(self.mt.hooks.values(), hook_idxs):
                hook.outputs[idx] = torch.cat([o for o in hook.outputs[idx:]], dim=1)
                hook.outputs = hook.outputs[:idx+1]
            # save matrix and utokens
            for h in SAVED_MODULES:
                matrixs = getattr(self, h+'_matrixs')
                cur_matrix = torch.cat([self.mt.hooks[f'{h}_{l}'].outputs[-1] for l in range(self.mt.n_layer)], dim=0)
                matrixs.append(cur_matrix)

                module_utokens = getattr(self, h+'_utokens')
                module_uprobs = getattr(self, h+'_uprobs')
                cur_uprob = torch.softmax(self.lm_head(self.ln_f(cur_matrix)), dim=-1)  # [n_layer, seq_len, vocab_size]
                
                # select utokens with top k diff
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
                        title_opts={"text": f"Unembedding Token Flow: {module}", 
                                    "subtext": f"Current Token: {cur_sentence[tidx]}"}
                    )
                )
                tl.add(bar, f"{l}")
            tab.add(tl, cur_sentence[tidx])
        return tab
    
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
        self.mt.reset_hook()