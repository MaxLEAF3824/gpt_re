import inspect
from symbol import raise_stmt
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
from torch import ones_like, optim, Tensor, zeros_like
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pytorch_lightning as pl


def camelize(string: str):
    return ''.join([i.capitalize() for i in string.split('_')])


class LLM(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_mt()
        self.configure_loss()

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None, labels: Tensor = None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids, attention_mask = batch[0], batch[1]
        return self(input_ids, attention_mask=attention_mask)

    def predict_next_token(self, input_texts):
        '''batch: list of str'''
        input_ids, attention_mask = self.tokenizer(input_texts, padding=True, return_tensors='pt').values()
        res = self.predict_step((input_ids, attention_mask), 0)
        pred_idxs = torch.argmax(res['logits'][:, -1, :], dim=1).unsqueeze(1)
        next_token = self.tokenizer.batch_decode(pred_idxs)
        return next_token

    def training_step(self, batch, batch_idx):
        '''batch: (input_ids, attention_mask, labels) **not padding** '''
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

        camel_opt_name = camelize(self.hparams.optimizer)
        if hasattr(optim, camel_opt_name):
            optimizer = getattr(optim, camel_opt_name)(
                self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0)
        try:
            lr_lambda = eval(self.hparams.lr_lambda)
        except:
            lr_lambda = None

        if lr_lambda is None:
            return optimizer
        else:
            scheduler = lrs.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return [optimizer], [scheduler]

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
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                # user fast tokenizer if possible
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = 'left'
            except:
                raise ValueError("illegal model name ")
