import contextlib
import inspect
from typing import Dict, List, Union
import torch
from torch import nn
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
from torch import ones_like, optim, Tensor, zeros_like
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pytorch_lightning as pl
from transformers.optimization import get_cosine_schedule_with_warmup
import os


def camelize(string: str):
    return ''.join([i.capitalize() for i in string.split('_')])


def recursive_copy(x, clone=None, detach=None, retain_grad=None, device=None):
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        if device:
            x = x.to(device)
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v, clone, detach, retain_grad, device) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v, clone, detach, retain_grad, device) for v in x])
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."


class LLMHook(Dict):
    def __init__(self,
                 module,
                 name=None,
                 retain_output=True,
                 retain_input=False,
                 edit_output=None,
                 clone=False,
                 detach=False,
                 device="cpu"):
        self.module = module
        self.name = name if name is not None else module._get_name()
        self.inputs = []
        self.outputs = []

        def hook(module, input, output):
            if retain_input:
                self.inputs.append(recursive_copy(input[0] if len(input) == 1 else input,
                                                  clone=clone, detach=detach, retain_grad=False, device=device))
            if retain_output:
                self.outputs.append(recursive_copy(output, clone=clone, detach=detach, device=device))
            if edit_output:
                output = edit_output(module, input, output)
            return output

        self.hook = module.register_forward_hook(hook)

    def remove(self):
        self.hook.remove()


class LLM(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_mt()
        self.configure_loss()
        self.hooks = {}

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None, labels: Tensor = None, **kwargs):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        input_ids, attention_mask = batch[0], batch[1]
        labels = batch[2] if len(batch) > 2 else None
        return self(input_ids, attention_mask=attention_mask, labels=labels)

    def set_predict_step(self, func):
        import types
        self.predict_step = types.MethodType(func, self)

    def predict_next_token(self, input_texts):
        '''batch: list of str'''
        input_ids, attention_mask = self.tokenizer(input_texts, padding=True, return_tensors='pt').values()
        res = self(input_ids, attention_mask=attention_mask)
        pred_idxs = torch.argmax(res['logits'][:, -1, :], dim=1).unsqueeze(1)
        next_token = self.tokenizer.batch_decode(pred_idxs)
        return next_token

    def training_step(self, batch, batch_idx):
        '''batch: (input_ids, attention_mask, labels) **padding already** '''
        input_ids, attention_mask, labels = batch

        res = self(input_ids, attention_mask=attention_mask, labels=labels)

        if isinstance(res.get('loss'), Tensor):
            loss = res['loss']
        else:
            lm_logits = res['logits']
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        '''batch: (input_ids, attention_mask, labels) **not padding**'''
        input_ids, attention_mask, labels = batch

        res = self(input_ids, attention_mask=attention_mask, labels=labels)

        lm_logits = res['logits']
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if isinstance(res.get('loss'), Tensor):
            loss = res['loss']
        else:
            loss = self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        pred = torch.argmax(shift_logits, dim=-1)  # [bsz, seq]

        total_num = torch.argwhere(shift_labels != -100).shape[0]
        correct_num = torch.sum(pred == shift_labels).cpu().item()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', correct_num/total_num, on_step=False, on_epoch=True, prog_bar=True)

        return (correct_num, total_num)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0

        # optimizer
        camel_opt_name = camelize(self.hparams.optimizer)
        if hasattr(optim, camel_opt_name):
            optimizer = getattr(optim, camel_opt_name)(
                self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0)

        # scheduler
        try:
            lr_lambda = eval(self.hparams.lr_lambda)
        except:
            lr_lambda = None

        if lr_lambda is not None:
            scheduler = lrs.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return [optimizer], [scheduler]
        elif lr_lambda == "cosine":
            warmup_t0 = self.hparams.warmup_t0
            scheduler = lrs.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_t0)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def generate(self, input_texts, **generate_kwargs):
        input_ids, attention_mask = self.tokenizer(input_texts, padding=True, return_tensors='pt').values()
        if 'max_new_tokens' not in generate_kwargs:
            generate_kwargs['max_new_tokens'] = 20
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        output_ids = self.model.generate(input_ids, attention_mask=attention_mask, **generate_kwargs)
        answer = self.tokenizer.batch_decode(output_ids)
        return answer

    def configure_loss(self):
        if not hasattr(self.hparams, "loss_func"):
            self.loss_func = F.cross_entropy
            return
        loss_func_name = self.hparams.loss_func.lower()
        if hasattr(F, loss_func_name):
            self.loss_func = getattr(F, loss_func_name)
        else:
            raise ValueError("illegal loss func")

    def load_mt(self):
        model_name = self.hparams.model_name
        camel_name = camelize(model_name)
        try:
            MT = getattr(importlib.import_module('.'+model_name, package=__package__), camel_name)
            mt = MT()
            self.model = mt.get_model()
            self.tokenizer = mt.get_tokenizer()
        except:
            try:
                # use cache dir as default to speed up
                cache_dir = os.environ.get('HF_HOME', None)
                cache_dir = os.path.join(cache_dir, 'hub') if cache_dir else None

                self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
                # user fast tokenizer if possible
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)
                if "llama" in model_name:
                    self.tokenizer.add_special_tokens(
                        {
                            "eos_token": "</s>",
                            "bos_token": "</s>",
                            "unk_token": "</s>",
                        }
                    )
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = 'left'
            except:
                raise ValueError("illegal model name ")

    def add_hook(self, module, name=None, retain_output=True, retain_input=False,
                 edit_output=None, clone=False, detach=False, device="cpu"):
        self.hooks[name] = LLMHook(module, name, retain_output, retain_input,
                                   edit_output, clone, detach, device)

    def clear_hook(self):
        for name, hook in self.hooks.items():
            hook.remove()
        self.hooks.clear()

    def save_hook(self, path=""):
        pass
