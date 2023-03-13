from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch.utils.data as tud
from torch.utils.data import DataLoader
import json
import os
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import random 
from dataclasses import dataclass
import dataclasses
import os
import wandb
os.environ["WANDB_MODE"] = "dryrun"
from collections import defaultdict
# train gpt2 for addition

@dataclass(frozen = True)
class Config:
    home_dir = '/home/max'    
    model_dir = f"{home_dir}/coding/huggingface_models/gpt2-math/small"
    bsz = 4096
    data_dir = "data"
    train_dataset_name = "train_2_zfill.txt"
    test_dataset_name = "test_2_zfill.txt"
    lr: float = 1e-4 #@param
    weight_decay: float = 1.0 #@param
    num_epochs: int = 5000 #@param
    save_every: int = 100 #@param
    save_models: bool = True #@param
    save_model_dir: str = os.path.join(model_dir,"saved_models") #@param
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    pth = None
    # pth = "/home/max/coding/huggingface_models/gpt2-math/small/saved_models/grok_1678627879/4700.pth"
    # device = t.device('cpu')
    print(f"device:{device}")
    def is_it_time_to_save(self, epoch):
        return (epoch % self.save_every == 0)

class AdditionDataset(tud.Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        with open(data_path, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        self.data = self.make_inputs(self.tokenizer, lines)

    def make_inputs(self, tokenizer, prompts):
        token_lists = [tokenizer.encode(p) for p in prompts]
        maxlen = max(len(t) for t in token_lists)
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
        eq = tokenizer.encode("=")[0]
        labels = []
        prompt_ids = []
        prompt_attention_mask = []
        for tok in token_lists:
            eq_idx = tok.index(eq)
            prompt_ids.append([pad_id] * (maxlen - len(tok[:eq_idx+1])) + tok[:eq_idx+1])
            prompt_attention_mask.append([0] * (maxlen - len(tok[:eq_idx+1])) + [1] * len(tok[:eq_idx+1]))
            labels.append([-100] * (maxlen - len(tok)) + [-100] * (eq_idx+1) + tok[eq_idx + 1:])
        input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
        attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
        return t.tensor([input_ids, attention_mask, labels, prompt_ids, prompt_attention_mask]).transpose(0,1)


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def gen_mt(config):
    model_dir = config.model_dir
    with open(os.path.join(model_dir,"model_config.json"), "r") as f:
        config_dict = json.load(f)

    config = GPT2Config.from_dict(config_dict)
    model = GPT2LMHeadModel(config)
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    return model, tokenizer

def gen_train_test(config: Config, tokenizer):
    '''Generate train and test split'''
    train = DataLoader(AdditionDataset(os.path.join(config.data_dir, config.train_dataset_name), tokenizer, ), batch_size=config.bsz, shuffle=True)
    test = DataLoader(AdditionDataset(os.path.join(config.data_dir, config.test_dataset_name), tokenizer, ), batch_size=config.bsz, shuffle=True)
                      
    return train, test   


class Trainer:
    '''TODO
    ways this stinks:
    - callbacks every k epochs 
    - training on infinite data
    - general abstract class w/o assumption and subclasses w/ more assumptions
    - check out hugging face trainer
    - disentangle optimization step and taking gradients
    - forward compatibility, e.g. batches per step
    '''

    def __init__(self, config : Config) -> None:
        wandb.init(project = "grokking", mode='offline', config = dataclasses.asdict(config))
        self.model, self.tokenizer = gen_mt(config)
        self.model.to(config.device)
        if getattr(config,"pth") is not None:
            self.model.load_state_dict(t.load(config.pth)['model'])
        self.optimizer = optim.AdamW(self.model.parameters(), lr = config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(step/10, 1)) # TODO make this a config option
        self.run_name = f"grok_{int(time.time())}"
        self.train, self.test = gen_train_test(config = config, tokenizer=self.tokenizer)
        print('training length = ', len(self.train))
        print('testing length = ', len(self.test))
        self.train_losses = []
        self.test_losses = []
        self.train_acces = []
        self.test_acces = []
        self.config = config

    def save_epoch(self, epoch, save_to_wandb = True):
        train_acc, wrong = self.eval_model(self.train)
        test_acc, wrong = self.eval_model(self.test)
        self.train_acces.append(train_acc)
        self.test_acces.append(test_acc)
        ''' precondition! train loss and test losses have been appended to '''
        save_dict = {
            'model': self.model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'train_acc': self.train_acces[-1],
            'test_acc': self.test_acces[-1],
            'epoch': epoch,
        }
        if save_to_wandb:
            wandb.log(save_dict)
            print("Saved epoch to wandb")
        if self.config.save_models: 
            t.save(save_dict, f"{self.config.save_model_dir}/{self.run_name}/{epoch}.pth")
            print(f"Saved model to {self.config.save_model_dir}/{self.run_name}/{epoch}.pth")

    def do_a_training_step(self, epoch: int):
        '''returns train_loss, test_loss'''
        train_loss = self.full_loss(data = self.train)
        test_loss = self.full_loss(data = self.test, backward=False)
        self.train_losses.append(train_loss.item())
        self.test_losses.append(test_loss.item())
        return train_loss, test_loss
    
    def full_loss(self, data, backward=True):
        '''Takes the cross entropy loss of the model on the data'''
        full_loss = t.tensor(0.0, device=self.config.device)
        bs = len(data)
        self.model.requires_grad_(backward)
        for d in data:
            input_ids = d[:,0,:].to(self.config.device)
            attention_mask = d[:,1,:].to(self.config.device)
            labels = d[:,2,:].to(self.config.device)
            res = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = res['loss']
            if backward:
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            full_loss += loss / bs
        return full_loss
    
    def eval_model(self, data):
        pad_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index("[PAD]")]
        acc = []
        wrong = []
        for d in data:
            prompt_ids = d[:,3,:].to(self.config.device)
            prompt_attn_mask = d[:,4,:].to(self.config.device)
            input_ids = d[:,0,:].to(self.config.device)
            labels = d[:,2,:].to(self.config.device)
            max_new_tokens = t.where(labels!=-100, t.ones_like(labels), t.zeros_like(labels)).sum(1).max().item()
            with t.no_grad():
                output = self.model.generate(inputs=prompt_ids, attention_mask=prompt_attn_mask, max_new_tokens=max_new_tokens)
            pred = self.tokenizer.batch_decode(output[:,-max_new_tokens:])
            gt = self.tokenizer.batch_decode(input_ids)
            gt = [g.split("=")[1].strip() for g in gt]
            for idx, (p, g) in enumerate(zip(pred, gt)):
                flag = True
                for i in range(len(g)):
                    if p[i] != g[i]:
                        flag = False
                        break
                acc.append(flag)
                if not flag:
                    wrong.append(self.tokenizer.decode(output[idx]))
        return np.mean(acc), wrong


    def initial_save_if_appropriate(self):
        if self.config.save_models:
            os.makedirs(f"{self.config.save_model_dir}/{self.run_name}")
            save_dict = {
                'model': self.model.state_dict(),
                'train_data' : self.train,
                'test_data' : self.test}
            t.save(save_dict, f"{self.config.save_model_dir}/{self.run_name}/init.pth")
    
    def post_training_save(self, save_optimizer_and_scheduler = True, log_to_wandb = True):
        if not self.config.save_models:
            os.makedirs(f"{self.config.save_model_dir}/{self.run_name}", exist_ok=True)
        save_dict = {
            'model': self.model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_acc': self.train_acces[-1],
            'test_acc': self.test_acces[-1],
            'train_acces': self.train_acces,
            'test_acces': self.test_acces,
            'epoch': self.config.num_epochs,
        }
        if save_optimizer_and_scheduler:
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['scheduler'] = self.scheduler.state_dict()
        if log_to_wandb:
            wandb.log(save_dict)
        t.save(save_dict, f"{self.config.save_model_dir}/{self.run_name}/final.pth")
        print(f"Saved model to {self.config.save_model_dir}/{self.run_name}final.pth")

def train_model(config: Config):
    world = Trainer(config = config)
    print(f'Run name {world.run_name}')
    world.initial_save_if_appropriate()

    for epoch in range(config.num_epochs):
        train_loss, test_loss = world.do_a_training_step(epoch)
        if config.is_it_time_to_save(epoch = epoch):
            world.save_epoch(epoch = epoch)
        print(f'Epoch {epoch}, train loss {t.log(train_loss.mean()).item():.4f}, test loss {t.log(test_loss.mean()).item():.4f}, train acc {world.train_acces[-1]:.4f}, test acc {world.test_acces[-1]:.4f}')
    world.post_training_save(save_optimizer_and_scheduler=True)
    return world # to export the dictionary with the training metrics

if __name__ == "__main__":
    train_model(config=Config())
