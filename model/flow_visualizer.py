from .model_interface import LLM
import torch
import os
from copy import deepcopy

SAVED_MODULES = ['layer', 'attn', 'mlp']


class FlowVisualizer:
    def __init__(self, mt: LLM, max_unembed_num=100):
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision('medium')
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self.max_unembed_num = max_unembed_num
        self.mt = mt.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.idx2token = {int(v): k for k, v in mt.tokenizer.get_vocab().items()}
        self.lm_head = deepcopy(mt.lm_head).to('cpu')
        self.init_save_hook()
        self.sentences = []
        self.layer_matrixs = []  # [n_layer, seq_len, hidden_size]
        self.attn_matrixs = []
        self.mlp_matrixs = []
        self.layer_unembed_tokens = [] # [n_layer, seq_len, max_unembed_num]
        self.attn_unembed_tokens = []
        self.mlp_unembed_tokens = []

        
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
            input_ids = input_ids.to(self.mt.device)
            attention_mask = attention_mask.to(self.mt.device)
        
            hook_idxs = [len(h.outputs) for h in self.mt.hooks.values()]
        
            output_ids = self.mt.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_wargs)
        
            # merge hook outputs
            for (hook, idx) in zip(self.mt.hooks.values(), hook_idxs):
                hook.outputs[idx] = torch.cat([o for o in hook.outputs[idx:]], dim=1)
                hook.outputs = hook.outputs[:idx+1]
            # save matrix and unembed_tokens
            for h in SAVED_MODULES:
                matrixs = getattr(self, h+'_matrixs')
                cur_matrix = torch.cat([self.mt.hooks[f'{h}_{l}'].outputs[-1] for l in range(self.mt.n_layer)], dim=0)
                matrixs.append(cur_matrix)

                unembed_tokens = getattr(self, h+'_unembed_tokens')
                cur_unembed_prob = torch.softmax(self.lm_head(cur_matrix), dim=-1)  # [n_layer, seq_len, vocab_size]
                cur_unembed_tokens_diff = (cur_unembed_prob[1:] - cur_unembed_prob[:-1]).abs().sum(dim=0) # [seq_len, vocab_size]
                cur_unembed_tokens_ids = torch.topk(cur_unembed_tokens_diff, k=self.max_unembed_num, dim=-1, sorted=True).indices # [seq_len, max_unembed_num]
                cur_unembed_prob = torch.index_select(cur_unembed_prob, dim=-1, index=cur_unembed_tokens_ids) # [n_layer, seq_len, max_unembed_num]
                
                unembed_tokens.append(cur_unembed_prob)
            out_tokens = self.mt.tokenizer.batch_decode(output_ids[0])
            self.sentences.append(out_tokens)
        self.mt.reset_hook()

    def visualize(self, idx=-1):
        pass
