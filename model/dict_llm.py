from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BertModel, AutoModel
from .llm_utils import print_struct
from .llm import LLM
from .llm_hooker import LLMHooker, LLMHookerConfig
import os
import json
import math
import numpy as np 
from typing import List, Dict
from functools import partial


def sinkhorn(dot, mask=None, eps=1e-03, return_kernel=False, max_iter=100):
    """
    dot: n x in_size x out_size
    mask: n x in_size
    output: n x in_size x out_size
    """
    n, in_size, out_size = dot.shape
    if return_kernel:
        K = torch.exp(dot / eps)
    else:
        K = dot
    # K: n x in_size x out_size
    u = K.new_ones((n, in_size))
    v = K.new_ones((n, out_size))
    a = float(out_size / in_size)
    if mask is not None:
        mask = mask.float()
        a = out_size / mask.sum(1, keepdim=True)
    for _ in range(max_iter):
        u = a / torch.bmm(K, v.view(n, out_size, 1)).view(n, in_size)
        if mask is not None:
            u = u * mask
        v = 1. / torch.bmm(u.view(n, 1, in_size), K).view(n, out_size)
    K = u.view(n, in_size, 1) * (K * v.view(n, 1, out_size))
    if return_kernel:
        K = K / out_size
        return (K * dot).sum(dim=[1, 2])
    return K

def log_sinkhorn(K, mask=None, eps=1.0, return_kernel=False, max_iter=100):
    """
    dot: n x in_size x out_size
    mask: n x in_size
    output: n x in_size x out_size
    """
    n, in_size, out_size = K.shape
    def min_eps(u, v, dim):
        Z = (K + u.view(n, in_size, 1) + v.view(n, 1, out_size)) / eps
        return -torch.logsumexp(Z, dim=dim)
    # K: n x in_size x out_size
    u = K.new_zeros((n, in_size))
    v = K.new_zeros((n, out_size))
    a = torch.ones_like(u).fill_(out_size / in_size)
    if mask is not None:
        a = out_size / mask.float().sum(1, keepdim=True)
    a = torch.log(a)
    for _ in range(max_iter):
        u = eps * (a + min_eps(u, v, dim=-1)) + u
        if mask is not None:
            u = u.masked_fill(~mask, -1e8)
        v = eps * min_eps(u, v, dim=1) + v
    if return_kernel:
        output = torch.exp(
            (K + u.view(n, in_size, 1) + v.view(n, 1, out_size)) / eps)
        output = output / out_size
        return (output * K).sum(dim=[1, 2])
    K = torch.exp(
        (K + u.view(n, in_size, 1) + v.view(n, 1, out_size)) / eps)
    return K

def multihead_attn(input, weight, mask=None, eps=1.0, return_kernel=False,
                   max_iter=100, log_domain=False, position_filter=None):
    """Comput the attention weight using Sinkhorn OT
    input: n x in_size x in_dim
    mask: n x in_size
    weight: m x out_size x in_dim (m: number of heads/ref)
    output: n x out_size x m x in_size
    """
    n, in_size, in_dim = input.shape
    m, out_size = weight.shape[:-1]
    K = torch.tensordot(input, weight, dims=[[-1], [-1]])
    K = K.permute(0, 2, 1, 3)
    if position_filter is not None:
        K = position_filter * K
    # K: n x m x in_size x out_size
    K = K.reshape(-1, in_size, out_size)
    # K: nm x in_size x out_size
    if mask is not None:
        mask = mask.repeat_interleave(m, dim=0)
    if log_domain:
        K = log_sinkhorn(K, mask, eps, return_kernel=return_kernel, max_iter=max_iter)
    else:
        if not return_kernel:
            K = torch.exp(K / eps)
        K = sinkhorn(K, mask, eps, return_kernel=return_kernel, max_iter=max_iter)
    # K: nm x in_size x out_size
    if return_kernel:
        return K.reshape(n, m)
    K = K.reshape(n, m, in_size, out_size)
    if position_filter is not None:
        K = position_filter * K
    K = K.permute(0, 3, 1, 2).contiguous()
    return K

class OTKernel(nn.Module):
    def __init__(self, in_dim, out_size, heads=1, eps=0.1, max_iter=100,
                 log_domain=False, position_encoding=None, position_sigma=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_size = out_size
        self.heads = heads
        self.eps = eps
        self.max_iter = max_iter

        self.weight = nn.Parameter(torch.Tensor(heads, out_size, in_dim))

        self.log_domain = log_domain
        self.position_encoding = position_encoding
        self.position_sigma = position_sigma

        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.out_size)
        for w in self.parameters():
            w.data.uniform_(-stdv, stdv)

    def get_position_filter(self, input, out_size):
        if input.ndim == 4:
            in_size1 = input.shape[1]
            in_size2 = input.shape[2]
            out_size = int(math.sqrt(out_size))
            if self.position_encoding is None:
                return self.position_encoding
            elif self.position_encoding == "gaussian":
                sigma = self.position_sigma
                a1 = torch.arange(1., in_size1 + 1.).view(-1, 1) / in_size1
                a2 = torch.arange(1., in_size2 + 1.).view(-1, 1) / in_size2
                b = torch.arange(1., out_size + 1.).view(1, -1) / out_size
                position_filter1 = torch.exp(-((a1 - b) / sigma) ** 2)
                position_filter2 = torch.exp(-((a2 - b) / sigma) ** 2)
                position_filter = position_filter1.view(
                    in_size1, 1, out_size, 1) * position_filter2.view(
                    1, in_size2, 1, out_size)
            if self.weight.is_cuda:
                position_filter = position_filter.cuda()
            return position_filter.reshape(1, 1, in_size1 * in_size2, out_size * out_size)
        in_size = input.shape[1]
        if self.position_encoding is None:
            return self.position_encoding
        elif self.position_encoding == "gaussian":
            # sigma = 1. / out_size
            sigma = self.position_sigma
            a = torch.arange(0., in_size).view(-1, 1) / in_size
            b = torch.arange(0., out_size).view(1, -1) / out_size
            position_filter = torch.exp(-((a - b) / sigma) ** 2)
        elif self.position_encoding == "hard":
            # sigma = 1. / out_size
            sigma = self.position_sigma
            a = torch.arange(0., in_size).view(-1, 1) / in_size
            b = torch.arange(0., out_size).view(1, -1) / out_size
            position_filter = torch.abs(a - b) < sigma
            position_filter = position_filter.float()
        else:
            raise ValueError("Unrecognizied position encoding")
        if self.weight.is_cuda:
            position_filter = position_filter.cuda()
        position_filter = position_filter.view(1, 1, in_size, out_size)
        return position_filter

    def get_attn(self, input, mask=None, position_filter=None):
        """Compute the attention weight using Sinkhorn OT
        input: batch_size x in_size x in_dim
        mask: batch_size x in_size
        self.weight: heads x out_size x in_dim
        output: batch_size x (out_size x heads) x in_size
        """
        return multihead_attn(
            input, self.weight, mask=mask, eps=self.eps,
            max_iter=self.max_iter, log_domain=self.log_domain,
            position_filter=position_filter)

    def forward(self, input, mask=None):
        """
        input: batch_size x in_size x in_dim
        output: batch_size x out_size x (heads x in_dim)
        """
        batch_size = input.shape[0]
        position_filter = self.get_position_filter(input, self.out_size)
        in_ndim = input.ndim
        if in_ndim == 4:
            input = input.view(batch_size, -1, self.in_dim)
        attn_weight = self.get_attn(input, mask, position_filter)
        # attn_weight: batch_size x out_size x heads x in_size

        output = torch.bmm(attn_weight.view(batch_size, self.out_size * self.heads, -1), input)
        if in_ndim == 4:
            out_size = int(math.sqrt(self.out_size))
            output = output.reshape(batch_size, out_size, out_size, -1)
        else:
            output = output.reshape(batch_size, self.out_size, -1)
        return output

class OTLayer(nn.Module):
    def __init__(self, in_dim, out_size, heads=1, eps=0.1, max_iter=10,
                 position_encoding=None, position_sigma=0.1, out_dim=None):
        super().__init__()
        self.out_size = out_size
        self.heads = heads
        if out_dim is None:
            out_dim = in_dim

        self.layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            OTKernel(out_dim, out_size, heads, eps, max_iter, log_domain=True,
                     position_encoding=position_encoding, position_sigma=position_sigma),
            )
        nn.init.xavier_uniform_(self.layer[0].weight)
        nn.init.xavier_uniform_(self.layer[2].weight)

    def forward(self, input):
        output = self.layer(input)
        return output

def no_mask(seq_len, spans):
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
    mask[:, :] = True
    return mask

def hierarchical_mask(seq_len, spans):
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
    for span in spans:
        sep_idx = span[-1]
        for i, j in span[:-1]:
            mask[i:j, i:j] = True
            mask[i:j, sep_idx] = True
            mask[sep_idx, i:j] = True
        mask[-1, sep_idx] = True
        mask[sep_idx, -1] = True
    return mask

def table_mask(seq_len, spans):
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
    sep_idxs = [0] + [span[-1] for span in spans]
    for sep_idx1, sep_idx2 in zip(sep_idxs[:-1], sep_idxs[1:]):
        mask[sep_idx1:sep_idx2, sep_idx1:sep_idx2] = True
        mask[sep_idx2, -1] = True
        mask[-1, sep_idx2] = True
    return mask

def no_position(seq_len, spans):
    return torch.zeros(seq_len, dtype=torch.long).reshape(-1)

def sequential_position(seq_len, spans):
    return torch.arange(seq_len, dtype=torch.long).reshape(-1)

def group_position(seq_len, spans):
    pos_ids = torch.zeros(seq_len, dtype=torch.long).reshape(-1)
    for span in spans:
        for i, j in span[:-1]:
            pos_ids[i:j] = torch.arange(j-i, dtype=torch.long) + 1
    return pos_ids

MASK_STRATEGIES = {
    "no": no_mask,
    "table": table_mask,
    "hierarchical": hierarchical_mask,
}

POSITION_STRATEGIES = {
    "no": no_position,
    "sequential": sequential_position,
    "group": group_position,
}

class DictEncoder(nn.Module):
    def __init__(
        self, 
        encoder_hidden_size,
        output_dim,
        num_encoder_head,
        num_encoder_layers,
        special_tokens_path,
        num_table_token,
        mask_strategy,
        position_strategy,
        encoder_type,
        mapper_type,
        max_length=4096,
        **kwargs
    ):
        super(DictEncoder, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.output_dim = output_dim
        self.num_encoder_head = num_encoder_head
        self.num_encoder_layers = num_encoder_layers
        self.special_tokens_path = special_tokens_path
        self.num_table_token = num_table_token
        self.mask_strategy = mask_strategy
        self.position_strategy = position_strategy
        self.encoder_type = encoder_type
        self.mapper_type = mapper_type
        self.max_length = max_length
        
        # init tok
        self.tok = AutoTokenizer.from_pretrained(os.path.join(os.environ['my_models_dir'], 'bert-base-chinese'))
        if special_tokens_path is not None:
            special_tokens = json.load(open(special_tokens_path))
            self.tok.add_tokens(special_tokens)

        # init encoder
        if encoder_type == 'transformer':
            self.embedding = nn.Embedding(len(self.tok), encoder_hidden_size)
            self.position_embedding = nn.Embedding(max_length, encoder_hidden_size)
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=encoder_hidden_size,
                    nhead=num_encoder_head,
                    batch_first=True,
                    dim_feedforward=encoder_hidden_size * 4,
                ),
                num_layers=num_encoder_layers,
                norm=nn.LayerNorm(encoder_hidden_size)
            )
        elif encoder_type == 'bert':
            # re_init bert embedding 
            self.bert = BertModel.from_pretrained(os.path.join(os.environ['my_models_dir'], 'bert-base-chinese'))
            if self.bert.embeddings.word_embeddings.num_embeddings != len(self.tok):
                self.bert.resize_token_embeddings(len(self.tok))
            self.bert.config.max_position_embeddings = max_length
            self.bert.embeddings.position_embeddings = nn.Embedding(self.bert.config.max_position_embeddings, self.bert.config.hidden_size)
            self.bert.embeddings.register_buffer("position_ids", torch.arange(self.bert.config.max_position_embeddings).expand((1, -1)), persistent=False)
            self.bert.embeddings.register_buffer("token_type_ids", torch.zeros(self.bert.embeddings.position_ids.size(), dtype=torch.long), persistent=False)
        
        # init mapper
        if mapper_type == 'otk':
            self.ot_layer = OTLayer(in_dim=encoder_hidden_size, out_size=num_table_token,heads=1,eps=0.1,max_iter=30,out_dim=output_dim)
        elif mapper_type == 'linear':
            self.linear1 = nn.Linear(encoder_hidden_size, output_dim)
            self.linear2 = nn.Linear(1, num_table_token)
        elif mapper_type == 'fc':
            self.fc = nn.Linear(encoder_hidden_size, output_dim * num_table_token)
        self.norm = nn.LayerNorm(output_dim)
    
    def prepare_batch_input(self, batch_dict: List[List[Dict]]):
        '''
        input_ids: torch.LongTensor with shape of (batch_size, seq_len)
        mask: torch.BoolTensor with shape of (batch_size, seq_len, seq_len)
        ATTENTION!!! mask is True for tokens that are **not masked**, False for tokens that are **masked**
        position_ids : torch.LongTensor with shape of (batch_size, seq_len)
        '''
        sep_id = self.tok.convert_tokens_to_ids('[SEP]')
        cls_id = self.tok.convert_tokens_to_ids('[CLS]')
        batch_size = len(batch_dict)
        batch_input_ids = []
        batch_mask = []
        batch_position_ids = []
        for dict in batch_dict:
            spans = [] # save the span of each key-value pair
            input_ids = []
            for d in dict:
                span = []
                for key, value in d['data'].items():
                    key_ids = self.tok(key, add_special_tokens=False)['input_ids']
                    value_ids = self.tok(value, add_special_tokens=False)['input_ids']
                    kv_ids = key_ids + value_ids
                    span.append((len(input_ids), len(input_ids)+len(kv_ids)))
                    input_ids.extend(kv_ids)
                span.append(len(input_ids))  # sep_idx
                input_ids.append(sep_id)
                spans.append(span)
            input_ids.append(cls_id)
            input_ids = torch.tensor(input_ids, dtype=torch.long).reshape(-1)
            seq_len = input_ids.shape[-1]

            mask = MASK_STRATEGIES.get(self.mask_strategy, no_mask)(seq_len, spans) # ATTENTION!!! True for tokens that are **not masked**, False for tokens that are **masked**
            position_ids = POSITION_STRATEGIES.get(self.position_strategy, no_position)(seq_len, spans)
            
            batch_input_ids.append(input_ids)
            batch_mask.append(mask)
            batch_position_ids.append(position_ids)
        
        max_len = max([input_ids.shape[-1] for input_ids in batch_input_ids])
        batch_input_ids = [F.pad(input_ids, (max_len - input_ids.shape[-1], 0), value=self.tok.pad_token_id).reshape(max_len) for input_ids in batch_input_ids]
        batch_mask = [F.pad(mask, (max_len - mask.shape[-1], 0, max_len - mask.shape[-1], 0), value=False).reshape(max_len, max_len) for mask in batch_mask]
        batch_position_ids = [F.pad(position_ids, (max_len - position_ids.shape[-1], 0), value=self.max_length - 2).reshape(max_len) for position_ids in batch_position_ids]
        
        input_ids = torch.vstack(batch_input_ids).reshape(batch_size, max_len)
        mask = torch.vstack(batch_mask).reshape(batch_size, max_len, max_len)
        position_ids = torch.vstack(batch_position_ids).reshape(batch_size, max_len)
        
        return input_ids, mask, position_ids, spans
    
    def encoder_batch_forward(self, input_ids, mask, position_ids):
        """Encoder Forward

        Args:
            input_ids (torch.LongTensor): (batch_size, seq_len)
            mask: torch.BoolTensor with shape of (batch_size, seq_len, seq_len)
            position_ids (torch.LongTensor): (batch_size, seq_len)

        Returns:
            output (torch.FloatTensor): (batch_size, seq_len, hidden_size)
        """
        if self.encoder_type == 'transformer':
            device = self.embedding.weight.device
            dtype = self.embedding.weight.data.dtype
            batch_size, seq_len = input_ids.shape
            input_ids = input_ids.to(device)
            position_ids = position_ids.to(device)
            attention_mask = torch.zeros_like(mask, dtype=dtype)
            NINF = torch.finfo(dtype).min
            attention_mask[~mask] = NINF
            attn_mask_3d = torch.repeat_interleave(attention_mask, self.num_encoder_head, dim=0).to(device)  # [N * H, T, S]
            src = (self.embedding(input_ids) + self.position_embedding(position_ids)).reshape(batch_size, seq_len, -1)
            output = self.transformer_encoder(src=src, mask=attn_mask_3d)
            return output
        elif self.encoder_type == 'bert':
            device = self.bert.embeddings.word_embeddings.weight.device
            dtype = self.bert.embeddings.word_embeddings.weight.data.dtype
            input_ids = input_ids.to(device)
            position_ids = position_ids.to(device)
            attention_mask = mask.to(dtype).to(device)
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)['last_hidden_state']
            return output
    
    def mapper_batch_forward(self, encoder_output, spans):
        """Mapper Forward

        Args:
            encoder_output (torch.FloatTensor): (batch_size, seq_len, hidden_size)

        Returns:
            output (torch.FloatTensor): (batch_size, self.num_table_token, self.output_dim)
        """
        if self.mapper_type == 'otk':
            sep_idxs = [span[-1] for span in spans]
            mapper_input = encoder_output[:, sep_idxs, :]
            output = self.ot_layer(mapper_input)
            return output
        elif self.mapper_type == 'linear':
            mapper_input = encoder_output[:, -1, :].unsqueeze(1) # [batch_size, hidden_size]
            output = self.linear1(mapper_input).transpose(1,2)
            output = self.linear2(output).transpose(1,2)
            output = output.reshape(-1, self.num_table_token, self.output_dim)
            return output
        elif self.mapper_type == 'fc':
            mapper_input = encoder_output[:, -1, :].unsqueeze(1)
            output = self.fc(mapper_input).reshape(-1, self.num_table_token, self.output_dim)
            return output
        
    def forward(self, batch_dict: List[List[Dict]]):
        input_ids, mask, position_ids, spans = self.prepare_batch_input(batch_dict)
        encoder_output = self.encoder_batch_forward(input_ids, mask, position_ids)
        output = self.mapper_batch_forward(encoder_output, spans)
        output = self.norm(output)
        return output

def full_edit_func(module, input_args, input_kwargs, output, batch_idxs, batch_embeddings: List[List]):
    hidden_states = output[0] if isinstance(output, tuple) else output
    batch_size, seq_len, _ = hidden_states.shape

    is_in_generation = (seq_len == 1)
    
    if is_in_generation:
        return output

    for i in range(batch_size):
        for j, table_idx in enumerate(batch_idxs[i]):
            sub_embedding = batch_embeddings[i][j]
            num_table_token = sub_embedding.shape[0]
            hidden_states[i, table_idx:table_idx+num_table_token, :] = sub_embedding
    
    return (hidden_states, *output[1:]) if isinstance(output, tuple) else hidden_states

class DictLLM(nn.Module):
    def __init__(
        self, 
        mt_path,
        encoder_hidden_size,
        num_table_token,
        num_encoder_head,
        num_encoder_layers,
        special_tokens_path,
        mask_strategy,
        position_strategy,
        encoder_type,
        mapper_type,
        deep_fusion,
        max_length
    ):
        super(DictLLM, self).__init__()
        self.num_table_token = num_table_token
        self.encoder_hidden_size = encoder_hidden_size
        self.deep_fusion = deep_fusion
        self.max_length = max_length
        self.fusion_layers = []
        self.llm = LLM.from_pretrained(mt_path=mt_path, torch_dtype=torch.float32)
        self.table_token = "[TABLE]"
        self.llm.tok.add_tokens([self.table_token])
        self.table_token_id = self.llm.tok.convert_tokens_to_ids(self.table_token)
        self.output_dim = self.llm.embedding.embedding_dim
        if self.deep_fusion:
            # self.fusion_layers = list(range(len(self.llm.block)))
            self.fusion_layers = [0,1,2,3]
        self.config = self.llm.model.config
        self.dict_encoder = DictEncoder(
            encoder_hidden_size=encoder_hidden_size,
            output_dim=self.output_dim,
            num_encoder_head=num_encoder_head,
            num_encoder_layers=num_encoder_layers,
            special_tokens_path=special_tokens_path,
            num_table_token=num_table_token*(len(self.fusion_layers) + 1),
            mask_strategy=mask_strategy,
            position_strategy=position_strategy,
            encoder_type=encoder_type,
            mapper_type=mapper_type,
        )

    def gradient_checkpointing_enable(self):
        self.llm.model.gradient_checkpointing_enable()
    
    def prepare_batch_input(self, batch_input_text : List[str], batch_dicts : List[List[List[Dict[str, str]]]] = None, batch_label_text : List[str] = None, cut_to_max_length : bool = False):
        # legal check
        batch_size = len(batch_input_text)
        device = self.llm.embedding.weight.data.device
        batch_dicts = [[] for i in range(batch_size)] if not batch_dicts else batch_dicts
        batch_num_stamps = [text.count(self.table_token) for text in batch_input_text]
        batch_num_dicts = [len(dicts) for dicts in batch_dicts]
        assert batch_num_stamps == batch_num_dicts, f"wrong dict num : {batch_num_stamps} table_token and {batch_num_dicts} dict"
        
        # input_ids, attention_mask     
        batch_input_text = [text.replace(self.table_token, "".join([self.table_token]*self.num_table_token)) for text in batch_input_text]
        inp = self.llm.tok(batch_input_text, padding=True, return_tensors='pt')
        input_ids, attention_mask = inp['input_ids'], inp['attention_mask']
        
        # labels
        labels = None
        if batch_label_text:
            batch_label_text = [t + f" {self.llm.tok.eos_token}" for t in batch_label_text]
            input_lens = attention_mask.sum(dim=1)
            all_text = [t1 + t2 for t1, t2 in zip(batch_input_text, batch_label_text)]
            inp = self.llm.tok(all_text, padding=True, return_tensors='pt')
            input_ids, attention_mask = inp['input_ids'], inp['attention_mask']
            all_lens = attention_mask.sum(dim=1)
            labels = torch.ones_like(input_ids) * -100
            for i in range(batch_size):
                labels[i, input_lens[i]:all_lens[i]] = input_ids[i, input_lens[i]:all_lens[i]]
        
        # batch_table_idxs
        batch_table_idxs = []
        for i in range(batch_size):
            batch_table_idxs.append([])
            for j in range(len(input_ids[i])-self.num_table_token):
                if torch.all(input_ids[i, j:j+self.num_table_token] == self.table_token_id):
                    batch_table_idxs[i].append(j)
        
        # replace table_token_id
        input_ids = input_ids.masked_fill(input_ids == self.table_token_id, self.llm.tok.pad_token_id)
        
        hook_configs = []
        # batch_dict_embeddings
        batch_dict_embeddings = [[] for i in range(batch_size)]
        all_dicts = [dict for dicts in batch_dicts for dict in dicts]
        if len(all_dicts) > 0:
            all_dict_embedding = self.dict_encoder(all_dicts).reshape((-1, self.num_table_token, len(self.fusion_layers) + 1, self.output_dim)).to(device)
            idx = 0
            for i, num_dicts in enumerate(batch_num_dicts):
                for j in range(num_dicts):
                    batch_dict_embeddings[i].append(all_dict_embedding[idx])
                    idx += 1
        
        # hook_configs
        # embedding_hook_config
        batch_emb_sub_embeddings = [[dict_embedding[:,0,:] for dict_embedding in dict_embeddings] for dict_embeddings in batch_dict_embeddings]
        hook_configs.append(LLMHookerConfig(
            "embedding", 
            save_output=False, 
            edit_output=partial(full_edit_func, batch_idxs=batch_table_idxs, batch_embeddings=batch_emb_sub_embeddings)
        ))
        
        # deep_fusion_hook_config
        for layer_idx in self.fusion_layers:
            batch_layer_sub_embeddings = [[dict_embedding[:,layer_idx+1,:] for dict_embedding in dict_embeddings] for dict_embeddings in batch_dict_embeddings]
            hook_configs.append(LLMHookerConfig(
                "block", 
                layer=layer_idx, 
                save_output=False, 
                edit_output=partial(full_edit_func, batch_idxs=batch_table_idxs, batch_embeddings=batch_layer_sub_embeddings)
            ))
        
        # cut to max_length and move to right device
        if cut_to_max_length:
            input_ids = input_ids[:,:self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
            if labels is not None:
                labels = labels[:,:self.max_length]
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        
        return input_ids, attention_mask, labels, hook_configs
        
    def forward(self, batch_input_text : List[str], batch_dicts : List[List[List[Dict[str, str]]]] = None, batch_label_text : List[str] = None, **kwargs):
        input_ids, attention_mask, labels, hook_configs = self.prepare_batch_input(batch_input_text, batch_dicts, batch_label_text, cut_to_max_length=True)

        with LLMHooker(self.llm, hook_configs):
            model_output = self.llm.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        return model_output
    
    @torch.no_grad()
    def generate(self, batch_input_text: List[str], batch_dicts: List[List[List[Dict[str, str]]]] = None, batch_label_text: List[str] = None, cut_input=False, **genkwargs):
        assert len(batch_input_text) == 1, "generate only support batch_size=1 now"
        
        input_ids, attention_mask, labels, hook_configs = self.prepare_batch_input(batch_input_text, batch_dicts=batch_dicts, batch_label_text=None)
        
        with LLMHooker(self.llm, hook_configs):
            output_ids = self.llm.model.generate(input_ids=input_ids, attention_mask=attention_mask, **genkwargs)
        
        return output_ids[:, input_ids.shape[-1]:] if cut_input else output_ids
