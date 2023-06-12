from typing import Dict, List, Union
import torch
from torch import nn
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
from torch import ones_like, optim, Tensor, zeros_like
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaModel
import pytorch_lightning as pl
import os
import types
import copy
import json
from .llm_hook import LLMHook
    
class LLM(pl.LightningModule):
    def __init__(self, **config):
        super().__init__()
        self.save_hyperparameters()

    @classmethod
    def from_pretrained(cls, **config):
        self = cls(**config)
        assert hasattr(self.hparams, "model_name"), "you should specify a model name when using from pretrained"
        self.model = AutoModelForCausalLM.from_pretrained(self.hparams.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name, use_fast=True)
        if getattr(self.hparams, 'fp16', False):
            self.model.half()
        self.post_init()
        return self
    
    @classmethod
    def from_mt(cls, model, tokenizer):
        self = cls()
        self.model = model
        self.tokenizer = tokenizer
        self.post_init()
        return self
    
    @classmethod
    def from_local(cls, **config):
        self = cls(**config)
        assert hasattr(self.hparams, "model_name"), "you should specify a model name when using from local"
        model_name = self.hparams.model_name
        camel_name = ''.join([i.capitalize() for i in model_name.split('_')])
        mt = getattr(importlib.import_module('.'+model_name, package=__package__), camel_name)()
        self.model = mt.model
        self.tokenizer = mt.tokenizer
        self.post_init()
        return self
    
    def post_init(self):
        self.tokenizer.add_special_tokens({
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
        })
        # add pad token and set padding side to the left
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        # configure loss function
        if not hasattr(self.hparams, "loss_func"):
            self.loss_func = F.cross_entropy
        else:
            loss_func_name = self.hparams.loss_func.lower()
            if hasattr(F, loss_func_name):
                self.loss_func = getattr(F, loss_func_name)
            else:
                raise ValueError("illegal loss func")
        
        # configure module config
        self.init_module_config(self.model)
        
    def init_module_config(self, model):
        model_class_name = model.__class__.__name__
        cwd = os.path.abspath(os.path.dirname(__file__))
        config_file_path = f'{cwd}/module_config/{model_class_name}.json'
        if not os.path.exists(config_file_path):
            raise ValueError(f"module config file not found, you should add a {model_class_name}.json in model/module_config/")
        config = json.load(open(config_file_path))
        self.module_config = config
        self.n_layer = eval(f"model.{config['n_layer']}")
        self.ln_f = eval(f"model.{config['ln_f_module']}")
        self.lm_head = eval(f"model.{config['lm_head_module']}")
        self.embedding = eval(f"model.{config['embedding_module']}")
        self.layers = []
        self.attns = []
        self.mlps = []
        self.ln_1s = []
        self.ln_2s = []
        for l in range(self.n_layer):
            self.layers.append(eval(f"model.{config['layer_module_tmp'].format(l)}"))
            self.attns.append(eval(f"model.{config['attn_module_tmp'].format(l)}"))
            self.mlps.append(eval(f"model.{config['mlp_module_tmp'].format(l)}"))
            self.ln_1s.append(eval(f"model.{config['ln_1_module_tmp'].format(l)}"))
            self.ln_2s.append(eval(f"model.{config['ln_2_module_tmp'].format(l)}"))
        self.hooks = {}
    
    def forward(self, input_ids: Tensor, attention_mask: Tensor = None, labels: Tensor = None, **kwargs):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        input_ids, attention_mask = batch[0], batch[1]
        labels = batch[2] if len(batch) > 2 else None
        return self(input_ids, attention_mask=attention_mask, labels=labels)

    def set_func(self, func_name, func):
        setattr(self, func_name, types.MethodType(func, self))

    def predict_next_token(self, input_texts):
        '''batch: str or list of str'''
        input_texts = [input_texts] if isinstance(input_texts, str) else input_texts
        tok_res = self.tokenizer(input_texts, padding=True, return_tensors='pt')
        input_ids, attention_mask = tok_res.input_ids, tok_res.attention_mask
        res = self(input_ids, attention_mask=attention_mask)
        pred_idxs = torch.argmax(res['logits'][:, -1, :], dim=1).unsqueeze(1)
        next_token = self.tokenizer.batch_decode(pred_idxs)
        return next_token

    def _acc(self, shift_logits, label):
        pred = torch.argmax(shift_logits, dim=-1)  # [bsz, seq]
        total_num = torch.argwhere(label != -100).shape[0]
        correct_num = torch.sum(pred == label).item()
        return correct_num / total_num

    def training_step(self, batch, batch_idx):
        '''batch: (input_ids, attention_mask, labels) **padding already** '''
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.unsqueeze(0) if len(
            input_ids.shape) == 1 else input_ids
        attention_mask = attention_mask.unsqueeze(0) if len(
            attention_mask.shape) == 1 else attention_mask
        labels = labels.unsqueeze(0) if len(labels.shape) == 1 else labels

        res = self(input_ids=input_ids,
                   attention_mask=attention_mask, labels=labels)

        lm_logits = res['logits']
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if isinstance(res.get('loss'), torch.Tensor):
            loss = res['loss']
        else:
            loss = self.loss_func(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        acc = self._acc(shift_logits, shift_labels)

        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False,
                 sync_dist=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        '''batch: (input_ids, attention_mask, labels) **not padding**'''
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.unsqueeze(0) if len(
            input_ids.shape) == 1 else input_ids
        attention_mask = attention_mask.unsqueeze(0) if len(
            attention_mask.shape) == 1 else attention_mask
        labels = labels.unsqueeze(0) if len(labels.shape) == 1 else labels

        res = self(input_ids=input_ids,
                   attention_mask=attention_mask, labels=labels)

        lm_logits = res['logits']
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if isinstance(res.get('loss'), Tensor):
            loss = res['loss']
        else:
            loss = self.loss_func(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        pred = torch.argmax(shift_logits, dim=-1)  # [bsz, seq]
        total_num = torch.argwhere(shift_labels != -100).shape[0]
        correct_num = torch.sum(pred == shift_labels).item()
        acc = self._acc(shift_logits, shift_labels)
        self.log('val_loss', loss, on_step=False,
                 on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False,
                 sync_dist=True, on_epoch=True, prog_bar=True)

        return (correct_num, total_num)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # optimizer
        lr = self.hparams.lr if hasattr(self.hparams, "lr") else 1e-4
        weight_decay = self.hparams.weight_decay if hasattr(
            self.hparams, 'weight_decay') else 1
        optimizer = optim.AdamW(self.trainer.model.parameters(
        ), lr=lr, weight_decay=weight_decay, fused=True)

        # scheduler
        try:
            lr_lambda = eval(self.hparams.lr_lambda)
        except:
            lr_lambda = None

        if lr_lambda is not None:
            scheduler = lrs.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return [optimizer], [scheduler]
        elif lr_lambda == "cosine":
            warmup_t0 = self.hparams.warmup_t0 if hasattr(
                self.hparams, 'warmup_t0') else 10
            scheduler = lrs.CosineAnnealingWarmRestarts(
                optimizer, T_0=warmup_t0)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def generate(self, input_texts, **generate_kwargs):
        tok_res = self.tokenizer(
            input_texts, padding=True, return_tensors='pt')
        input_ids, attention_mask = tok_res.input_ids, tok_res.attention_mask
        if 'max_new_tokens' not in generate_kwargs:
            generate_kwargs['max_new_tokens'] = 20
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        output_ids = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        answer = self.tokenizer.batch_decode(output_ids)
        return answer

    def add_hook(self, module, name=None, **kw_args):
        self.hooks[name] = LLMHook(module, name, **kw_args)

    def clear_hook(self):
        for name, hook in self.hooks.items():
            hook.outputs.clear()
            hook.inputs.clear()
            hook.remove()
        self.hooks.clear()

    def reset_hook(self):
        for name, hook in self.hooks.items():
            hook.outputs.clear()
            hook.inputs.clear()
