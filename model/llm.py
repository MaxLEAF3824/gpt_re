import os
import json
import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from .llm_hooker import LLMHooker, LLMHookerConfig
from .llm_utils import LoadWoInit

class Unembedding(nn.Module):
    def __init__(self, lm_head, ln_f):
        super().__init__()
        self.lm_head = lm_head
        self.ln_f = ln_f
        self.requires_grad_(False)
        
    def forward(self, x):
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x

class LLM(nn.Module):

    def __init__(self, **config):
        super().__init__()
        self.config=config

    @classmethod
    def from_pretrained(cls, **config):
        self = cls(**config)
        mt_path = self.config.get("mt_path", None)
        use_flash_attention_2 = self.config.get("use_flash_attention_2", False)
        torch_dtype = self.config.get("torch_dtype", torch.float16)
        with LoadWoInit():
            self.model = AutoModelForCausalLM.from_pretrained(mt_path, trust_remote_code=True, torch_dtype=torch_dtype,use_flash_attention_2=use_flash_attention_2)
        self.tok = AutoTokenizer.from_pretrained(mt_path, trust_remote_code=True, use_fast=True)
        self.post_init()
        return self
    
    @classmethod
    def from_mt(cls, model, tok, **config):
        self = cls(**config)
        self.model = model
        self.tok = tok
        self.post_init()
        return self
    
    def post_init(self):
        if not self.tok.eos_token:
            self.tok.add_special_tokens({"eos_token": "</s>"})
        if not self.tok.bos_token:
            self.tok.add_special_tokens({"bos_token": "<s>"})
        if not self.tok.unk_token:
            self.tok.add_special_tokens({"unk_token": "<unk>"})
        if not self.tok.pad_token:
            self.tok.pad_token = self.tok.eos_token
        
        # configure module config
        model_class_name = self.model.__class__.__name__
        model = self.model
        cwd = os.path.abspath(os.path.dirname(__file__))
        config_file_path = f'{cwd}/module_config/{model_class_name}.json'
        if not os.path.exists(config_file_path):
            raise ValueError(f"module config file not found, you should add a {model_class_name}.json in model/module_config/")
        config = json.load(open(config_file_path))
        self.module_config = config
        self.idx2token = [f"{i}-{self.tok.decode(i)}" for i in range(self.tok.vocab_size)]
        self.n_layer = eval(f"model.{config['n_layer']}")
    
        self.ln_f = eval(f"model.{config['ln_f_module']}")
        self.lm_head = eval(f"model.{config['lm_head_module']}")
        self.embedding = eval(f"model.{config['embedding_module']}")
        self.block = []
        self.attn = []
        self.mlp = []
        self.ln_1 = []
        self.ln_2 = []
        for l in range(self.n_layer):
            self.block.append(eval(f"model.{config['block_module_tmp'].format(l)}"))
            self.attn.append(eval(f"model.{config['attn_module_tmp'].format(l)}"))
            self.mlp.append(eval(f"model.{config['mlp_module_tmp'].format(l)}"))
            self.ln_1.append(eval(f"model.{config['ln_1_module_tmp'].format(l)}"))
            self.ln_2.append(eval(f"model.{config['ln_2_module_tmp'].format(l)}"))
          
    def forward(self, **kwargs):
        return self.model(**kwargs)

    def generate(self, input_texts, cut_input=False, **generate_kwargs):
        inp = self.tok(input_texts, padding=True, return_tensors='pt')
        input_ids, attention_mask = inp['input_ids'], inp['attention_mask']
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        output_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        if cut_input:
            output_ids = output_ids[:, input_ids.shape[-1]:]
        output_text = self.tok.batch_decode(output_ids)
        return output_text
    
    def vis_sentence(self, input_text, show_modules=['block','attn','mlp'], utokens_num=20, show_diff=True, **gen_wargs):
        from pyecharts import options as opts
        from pyecharts.charts import Bar, Timeline, Tab, Line
        unembedding = Unembedding(deepcopy(self.lm_head).to('cpu').float(), deepcopy(self.ln_f).to('cpu').float())
        inp = self.tok(input_text, return_tensors='pt')
        hook_configs = [LLMHookerConfig(module_name=m,layer=l, float=True, detach=True, save_input=(l == 0 and m == 'block')) for l in range(self.n_layer) for m in show_modules]
        num_modules = len(show_modules)
        with torch.no_grad(), LLMHooker(self, hook_configs) as hooker:
            gen_wargs['max_new_tokens'] = 1 if 'max_new_tokens' not in gen_wargs else gen_wargs['max_new_tokens']
            output_ids = self.model.generate(input_ids=inp['input_ids'].to(self.model.device), attention_mask=inp['attention_mask'].to(self.model.device), **gen_wargs)
            
            # 模型会在generate的过程中多次forward产生多个hook中间值，需要把hook的输出拼接起来得到完整的句子的matrix
            for hook in hooker.hooks:
                hook.outputs = [torch.cat([o for o in hook.outputs], dim=1)]
                if hook.config.save_input:
                    hook.inputs= [torch.cat([o for o in hook.inputs], dim=1)]
            
            # 保存generate的句子和下一个token
            out_tokens = self.tok.batch_decode(output_ids[0])
            cur_sentence = out_tokens[:-1]
            
            # 获取当前句子的关于每一层，每一个模块合并后的完整matrix [num_modules, n_layer, seq_len, hidden_size]
            cur_matrix = []
            for m in show_modules:
                module_hooks = sorted([h for h in hooker.hooks if h.config.module_name == m], key=lambda h: h.config.layer)
                cur_matrix.append(torch.cat([h.outputs[0] for h in module_hooks], dim=0))
            cur_matrix = torch.stack(cur_matrix)
            seq_len = cur_matrix.shape[2]
            
            # 将activation映射到vocabulary词表空间，计算所有unbedding token的概率
            cur_logits = unembedding(cur_matrix) # [num_modules, n_layer, seq_len, vocab_size]
            cur_prob = torch.softmax(cur_logits, dim=-1)  # [num_modules, n_layer, seq_len, vocab_size]

            # 计算层信息熵
            cur_info = -torch.sum(cur_prob * torch.log(cur_prob), dim=-1) # [num_modules, n_layer, seq_len]

            # 计算层概率差
            block_hook0 = [h for h in hooker.hooks if h.config.module_name == 'block' and h.config.layer == 0][0]
            x0 = block_hook0.inputs[0]# [1, seq_len, hidden_size]
            logits0 = unembedding(x0.unsqueeze(0).repeat(num_modules,1,1,1)) # [num_modules, 1, seq_len, vocab_size]
            cur_logits_extended = torch.cat([logits0, cur_logits], dim=1) # [num_modules, n_layer+1, seq_len, vocab_size]
            cur_diff = F.cross_entropy(cur_logits_extended[:,:-1].reshape(-1, cur_logits_extended.shape[-1]), cur_prob.reshape(-1, cur_prob.shape[-1]), reduction='none') # [num_modules * n_layer * seq_len]
            cur_diff = cur_diff.reshape(num_modules, self.n_layer, seq_len) # [num_modules, n_layer, seq_len]
            
            # 对generate的句子的每一个token对应的uprob，依据uprob在3个模块中的变化大小之和，对utoken从大到小排序
            cur_utokens = [] # [seq_len, vocab_size]
            cur_uprobs = [] # [seq_len, num_modules, n_layer, vocab_size]
            for j in range(seq_len):
                cur_token_prob = cur_prob[:,:,j,:] # [num_modules, n_layer, vocab_size]
                # 计算token在num_modules个模块中的概率变化之和，【这一行是速度瓶颈】
                cur_token_prob_diff = (cur_token_prob[1:] - cur_token_prob[:-1]).abs().sum(dim=0).sum(dim=0) # [vocab_size]
                # 按照变化之和从大到小排序
                cur_token_udiff, cur_token_uids = torch.topk(cur_token_prob_diff, k=utokens_num)
                cur_token_utokens = [self.idx2token[idx] for idx in cur_token_uids]
                cur_utokens.append(cur_token_utokens)
                cur_token_uprobs = cur_token_prob[:, :, cur_token_uids] # [num_modules, n_layer, vocab_size]
                cur_uprobs.append(cur_token_uprobs)
            
            cur_uprobs = torch.stack(cur_uprobs).transpose(0, 1) # [num_modules, seq_len, n_layer, vocab_size]
            
        # visualize utokens
        utokens_tab = Tab()
        for tidx in range(len(cur_sentence)):
            tl = Timeline()
            for l in range(self.n_layer):
                cur_utokens_ = cur_utokens[tidx][:utokens_num]
                cur_uprobs_ = cur_uprobs[:,tidx,l,:utokens_num] # [num_modules, utokens_num]
                bar = Bar()
                bar = bar.add_xaxis(cur_utokens_)
                for i,m in enumerate(show_modules):
                    bar = bar.add_yaxis(m, cur_uprobs_[i].numpy().tolist(), label_opts=opts.LabelOpts(is_show=False))
                bar = bar.reversal_axis()
                bar = bar.set_global_opts(
                        title_opts={"text": f"Unembedding Token Flow"},
                        xaxis_opts=opts.AxisOpts(name="Probability"),
                        yaxis_opts=opts.AxisOpts(name="Top k UTokens"),
                    )
                tl.add(bar, f"{l+1}")
            utokens_tab.add(tl, cur_sentence[tidx])
        
        # visualize entropy
        entropy_tab = Tab()
        for tidx in range(len(cur_sentence)):
            cur_info_ = cur_info[:,:,tidx] # [num_modules, n_layer]
            cur_diff_ = cur_diff[:,:,tidx] # [num_modules, n_layer]
            xaxis = [str(l+1) for l in list(range(self.n_layer))]
            line = Line()
            line = line.add_xaxis(xaxis)
            line = line.extend_axis(yaxis=opts.AxisOpts(name="Cross Entropy", type_="value", position="right"))
            line = line.extend_axis(yaxis=opts.AxisOpts(name="Infomation Entropy", type_="value", position="left"))
            for i,m in enumerate(show_modules):
                line = line.add_yaxis(f"{m} entropy", cur_info_[i].numpy().tolist(), yaxis_index=0, label_opts=opts.LabelOpts(is_show=False))
                if show_diff:
                    line = line.add_yaxis(f"{m} cross_entropy", cur_diff_[i].numpy().tolist(), yaxis_index=1, label_opts=opts.LabelOpts(is_show=False))
            line = line.set_global_opts(
                    title_opts=opts.TitleOpts(title="信息熵和交叉熵"),
                    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"))
            entropy_tab.add(line, cur_sentence[tidx])
            
        return utokens_tab, entropy_tab
    
    def get_similar_token(self, token_id, k=20):
        embedding = self.embedding.weight.data
        with torch.no_grad():
            cos_values, cos_indices = torch.topk(torch.cosine_similarity(embedding, embedding[token_id].unsqueeze(0), dim=1),k=k)
        return [f"{self.idx2token[id]}: {cos_values[i].item():.3f}" for i, id in enumerate(cos_indices)]