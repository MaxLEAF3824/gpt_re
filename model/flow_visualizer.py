from .model_interface import LLM
import torch
from copy import deepcopy
from pyecharts import options as opts
from pyecharts.charts import Bar, Timeline, Tab, Page, Line
from pyecharts.faker import Faker
import ipywidgets as widgets
from IPython.display import display
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
SAVED_MODULES = ['layer', 'attn', 'mlp']


class Unembedding(nn.Module):
    def __init__(self, lm_head, ln_f):
        super().__init__()
        self.lm_head = lm_head
        self.ln_f = ln_f
        
    def forward(self, x):
        with torch.no_grad():
            x = self.ln_f(x)
            x = self.lm_head(x)
        return x

class FlowVisualizer:
    def __init__(self, mt: LLM):
        self.mt = mt
        self.idx2token = [f"{i}-{self.mt.tokenizer.decode(i)}" for i in range(self.mt.tokenizer.vocab_size)]
        self.unembedding = Unembedding(deepcopy(mt.lm_head).to('cpu').float(), deepcopy(mt.ln_f).to('cpu').float())
        self.init_save_hook()
        self.sentences = [] # generated sentences
        self.next_tokens = [] # next token of sentences
        self.prompt_lengths = [] # prompt length of sentences
        self.utokens = [] # 对于每个句子，都有seq_len个token，每个token都有一个vocab_size大小的utoken list [bsz, seq_len, vocab_size]
        self.uprobs = [] # [bsz, 3, seq_len, n_layer, vocab_size]
        self.infos = [] # 对于每个句子，每个模块每一层每个token的uprob信息熵 [bsz, 3, n_layer, seq_len]
        self.diffs = [] # 对于每个句子，每个模块每一层每个token的uprob关于上一层的uprob的交叉熵 [bsz, 3, n_layer, seq_len]
        
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
        for h in SAVED_MODULES:
            for l in range(self.mt.n_layer):
                hook_config['retain_input'] = (l == 0 and h == 'layer') # 只保留Layer第一层的输入
                self.mt.add_hook(module=getattr(self.mt, h+'s')[l], name=f'{h}_{l}', **hook_config)

    def get_sentence_matrix(self, sidx):
        '''return matrix of sentence sidx with shape of [3, n_layer, seq_len, hidden_size]'''
        cur_matrix = torch.stack([torch.cat([self.mt.hooks[f'{h}_{l}'].outputs[sidx] for l in range(self.mt.n_layer)], dim=0) for h in SAVED_MODULES])
        return cur_matrix
    
    def get_x0(self, sidx):
        return self.mt.hooks['layer_0'].inputs[sidx]# [1, seq_len, hidden_size]
    
    def generate(self, input_texts, **gen_wargs):
        input_texts = input_texts if isinstance(input_texts, list) else [input_texts]
        inps = [self.mt.tokenizer(text, return_tensors='pt') for text in input_texts]
        
        for inp in tqdm(inps, total=len(inps)):
            input_ids, attention_mask = inp['input_ids'], inp['attention_mask']
            self.prompt_lengths.append(input_ids.shape[1])

            # model generate
            hook_idxs = [len(h.outputs) for h in self.mt.hooks.values()]
            with torch.no_grad():
                input_ids = input_ids.to(self.mt.model.device)
                attention_mask = attention_mask.to(self.mt.model.device)
                gen_wargs['max_new_tokens'] = 10 if 'max_new_tokens' not in gen_wargs else gen_wargs['max_new_tokens']
                output_ids = self.mt.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_wargs)
            
            # 模型会在generate的过程中多次forward产生多个hook中间值，需要把hook的输出拼接起来得到完整的句子的matrix
            for (hook, idx) in zip(self.mt.hooks.values(), hook_idxs):
                hook.outputs[idx] = torch.cat([o for o in hook.outputs[idx:]], dim=1)
                hook.outputs = hook.outputs[:idx+1]
                if hook.retain_input:
                    hook.inputs[idx] = torch.cat([o for o in hook.inputs[idx:]], dim=1)
                    hook.inputs = hook.inputs[:idx+1]
            
            # 保存generate的句子和下一个token
            out_tokens = self.mt.tokenizer.batch_decode(output_ids[0])
            self.sentences.append(out_tokens[:-1])
            self.next_tokens.append(out_tokens[-1])
            
            # 获取当前句子的关于每一层，每一个模块合并后的完整matrix [3, n_layer, seq_len, hidden_size]
            cur_matrix = self.get_sentence_matrix(-1)
            seq_len = cur_matrix.shape[2]
            
            # 将activation映射到vocabulary词表空间，计算所有unbedding token的概率
            cur_matrix[1] = cur_matrix[0]+cur_matrix[1] #  attn+layer
            cur_logits = self.unembedding(cur_matrix) # [3, n_layer, seq_len, vocab_size]
            cur_prob = torch.softmax(cur_logits, dim=-1)  # [3, n_layer, seq_len, vocab_size]

            # 计算层信息熵
            cur_info = -torch.sum(cur_prob * torch.log(cur_prob), dim=-1) # [3, n_layer, seq_len]
            self.infos.append(cur_info)

            # 计算层概率差
            x0 = self.get_x0(-1) # [1, seq_len, hidden_size]
            logits0 = self.unembedding(x0.unsqueeze(0).repeat(3,1,1,1)) # [3, 1, seq_len, vocab_size]
            cur_logits_extended = torch.cat([logits0, cur_logits], dim=1) # [3, n_layer+1, seq_len, vocab_size]
            cur_diff = F.cross_entropy(cur_logits_extended[:,:-1].reshape(-1, cur_logits_extended.shape[-1]), cur_prob.reshape(-1, cur_prob.shape[-1]), reduction='none') # [3 * n_layer * seq_len]
            cur_diff = cur_diff.reshape(3, self.mt.n_layer, seq_len) # [3, n_layer, seq_len]
            self.diffs.append(cur_diff)
            
            # 对generate的句子的每一个token对应的uprob，依据uprob在3个模块中的变化大小之和，对utoken从大到小排序
            cur_utokens = [] # [seq_len, vocab_size]
            cur_uprobs = [] # [seq_len, 3, n_layer, vocab_size]
            for j in range(seq_len):
                cur_token_prob = cur_prob[:,:,j,:] # [3, n_layer, vocab_size]
                # 计算token在3个模块中的概率变化之和
                cur_token_prob_diff = (cur_token_prob[1:] - cur_token_prob[:-1]).abs().sum(dim=0).sum(dim=0) # [vocab_size]
                # 按照变化之和从大到小排序
                cur_token_udiff, cur_token_uids = torch.sort(cur_token_prob_diff, descending=True)
                cur_token_utokens = [self.idx2token[idx] for idx in cur_token_uids]
                cur_utokens.append(cur_token_utokens)
                cur_token_uprobs = cur_token_prob[:, :, cur_token_uids] # [3, n_layer, vocab_size]
                cur_uprobs.append(cur_token_uprobs)
            
            # 保存utokens和uprobs
            self.utokens.append(cur_utokens)
            cur_uprobs = torch.stack(cur_uprobs).transpose(0, 1) # [3, seq_len, n_layer, vocab_size]
            self.uprobs.append(cur_uprobs)

    def visualize_utokens(self, sidx=-1, unum=20):
        cur_sentence = self.sentences[sidx]
        tab = Tab()
        for tidx in range(len(cur_sentence)):
            tl = Timeline()
            for l in range(self.mt.n_layer):
                cur_utokens = self.utokens[sidx][tidx][:unum]
                cur_uprobs = self.uprobs[sidx][:,tidx,l,:unum] # [3, unum]
                bar = (
                    Bar()
                    .add_xaxis(cur_utokens)
                    .add_yaxis('layer', cur_uprobs[0].numpy().tolist(), label_opts=opts.LabelOpts(is_show=False))
                    .add_yaxis('attn', cur_uprobs[1].numpy().tolist(), label_opts=opts.LabelOpts(is_show=False))
                    .add_yaxis('mlp', cur_uprobs[2].numpy().tolist(), label_opts=opts.LabelOpts(is_show=False))
                    .reversal_axis()
                    .set_global_opts(
                        title_opts={"text": f"Unembedding Token Flow"},
                        xaxis_opts=opts.AxisOpts(name="Probability"),
                        yaxis_opts=opts.AxisOpts(name="Top k UTokens"),
                    )
                )
                tl.add(bar, f"{l+1}")
            tab.add(tl, cur_sentence[tidx])
        return tab
    
    def visualize_info(self, sidx=-1, show_modules=['layer', 'attn', 'mlp'],show_diff=True):
        cur_sentence = self.sentences[sidx]
        tab = Tab()
        for tidx in range(len(cur_sentence)):
            cur_info = self.infos[sidx][:,:,tidx] # [3, n_layer]
            cur_diff = self.diffs[sidx][:,:,tidx] # [3, n_layer]
            xaxis = [str(l+1) for l in list(range(self.mt.n_layer))]
            c = (
                Line()
                .add_xaxis(xaxis)
                .extend_axis(
                    yaxis=opts.AxisOpts(
                        name="Cross Entropy",
                        type_="value",
                        position="right",
                    )
                )
                .extend_axis(
                    yaxis=opts.AxisOpts(
                        name="Infomation Entropy",
                        type_="value",
                        position="left",
                    )
                )
                .add_yaxis("layer info", cur_info[0].numpy().tolist(), yaxis_index=0, label_opts=opts.LabelOpts(is_show=False),)
                .set_series_opts(yaxis_opts=opts.AxisOpts(is_show=False))
                .add_yaxis("attn info", cur_info[1].numpy().tolist(), yaxis_index=0, label_opts=opts.LabelOpts(is_show=False))
                .add_yaxis("mlp info", cur_info[2].numpy().tolist(), yaxis_index=0, label_opts=opts.LabelOpts(is_show=False))
                .add_yaxis("layer diff", cur_diff[0].numpy().tolist(), yaxis_index=1, label_opts=opts.LabelOpts(is_show=False))
                .add_yaxis("attn diff", cur_diff[1].numpy().tolist(), yaxis_index=1, label_opts=opts.LabelOpts(is_show=False))
                .add_yaxis("mlp diff", cur_diff[2].numpy().tolist(), yaxis_index=1, label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="信息熵和交叉熵"),
                    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),)
            )
            tab.add(c, cur_sentence[tidx])
        return tab
    
    def get_similar_token(self, token_id, k=20):
        embedding = self.mt.embedding.weight.data
        with torch.no_grad():
            cos_values, cos_indices = torch.topk(torch.cosine_similarity(embedding, embedding[token_id].unsqueeze(0), dim=1),k=k)
        return [f"{self.idx2token[id]}: {cos_values[i].item():.3f}" for i, id in enumerate(cos_indices)]
        
    def clear(self):
        self.sentences.clear()
        self.next_tokens.clear()
        self.prompt_lengths.clear()
        self.utokens.clear() 
        self.uprobs.clear() 
        self.infos.clear()
        self.diffs.clear()