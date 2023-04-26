# %% [markdown]
# 初始化

# %%
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
import time
import pytorch_lightning as pl
import torch
from model.model_interface import LLM
import torch.utils.data as tud
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from tqdm.notebook import tqdm
from utils.my_utils import *
import torch.nn.functional as F
import random
import regex as re
from dataset import *
import ipywidgets as widgets
from IPython.display import display


torch.set_float32_matmul_precision('medium')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Global variables
global mt
global n_layer
global lm_head
global embedding
global ln_f
global blocks
global attn_name
global mlp_name
global trainer
global device_idxs
device_idxs = [1, 4]

model_list = list({
    "gpt2": "/nvme/guoyiqiu/coding/huggingface/hub/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
    "gpt2_xl": "/nvme/guoyiqiu/coding/huggingface/hub/models--gpt2-xl/snapshots/33cdb5c0db5423c1879b1b9f16c352988e8754a8",
    "llama_7b": "/nvme/share/guoyiqiu/llama-7b",
    "llama_13b": "/nvme/share/guoyiqiu/llama-13b",
    "vicuna_7b": "/nvme/share/guoyiqiu/vicuna-7b",
    "vicuna_13b": "/nvme/share/guoyiqiu/vicuna-13b-v1.1",
}.items())

llm_config = {
    "optimizer": "adamw",
    "lr": 1e-4,
}

hook_config = {
    "retain_output": True,
    "retain_input": False,
    "retain_grad": False,
    "edit_output": None,
    "clone": True,
    "float": True,
    "detach": True,
    "device": "cpu"
}

def init_mt():
    global mt
    mt = LLM(model_name=mt_dropdown.value, **llm_config)


def init_modules():
    global n_layer
    global lm_head
    global embedding
    global ln_f
    global blocks
    global attn_name
    global mlp_name
    if "gpt2" in mt.model.__class__.__name__.lower():
        # gpt2 config
        n_layer = mt.model.config.num_hidden_layers
        lm_head = mt.model.lm_head
        embedding = mt.model.transformer.wte
        ln_f = mt.model.transformer.ln_f
        blocks = mt.model.transformer.h
        attn_name = 'attn'
        mlp_name = 'mlp'
    elif "llama" in mt.model.__class__.__name__.lower():
        # llama config
        n_layer = mt.model.config.num_hidden_layers
        lm_head = mt.model.lm_head
        embedding = mt.model.model.embed_tokens
        ln_f = mt.model.model.norm
        blocks = mt.model.model.layers
        attn_name = 'self_attn'
        mlp_name = 'mlp'


def init_hook(mt):
    mt.clear_hook()
    for i in range(n_layer):
        mt.add_hook(module=blocks[i], name=f"block_{i}", **hook_config)
        mt.add_hook(getattr(blocks[i], attn_name), name=f"attn_{i}", **hook_config)
        mt.add_hook(getattr(blocks[i], mlp_name), name=f"mlp_{i}", **hook_config)


def setup(btn):
    time_st = time.time()
    btn.description = "Loading model..."
    init_mt()
    btn.description = "init modules..."
    init_modules()
    btn.description = "init hooks..."
    init_hook(mt)
    btn.description = "Everything is ready."
    device_tbtn.value = 'cpu'
    precision_tbtn.value = 'float'
    print(f"Time cost: {time.time() - time_st:.2f}s")

# setup widgets


# model dropdown
mt_dropdown = widgets.Dropdown(
    options=model_list,
    description='Model:',
    disabled=False,
)

# setup button
setup_btn = widgets.Button(
    description="Setup everything",
    disabled=False,
)
setup_btn.on_click(setup)

# switch deivce
device_tbtn = widgets.ToggleButtons(
    options=['cpu', f'cuda:{device_idxs[0]}',],
    disabled=False,
)


def switch_device(change):
    device_tbtn.disabled = True
    mt.model.to(change.new)
    torch.cuda.empty_cache() if change.new == 'cpu' else None
    device_tbtn.disabled = False


device_tbtn.observe(switch_device, names='value')

# switch precision
precision_tbtn = widgets.ToggleButtons(
    options=['float', 'half'],
    disabled=False,
)


def switch_precision(change):
    precision_tbtn.disabled = True
    if change.new == 'float':
        mt.model = mt.model.float()
        init_modules()
    elif change.new == 'half':
        mt.model = mt.model.half()
        init_modules()
    precision_tbtn.disabled = False


precision_tbtn.observe(switch_precision, names='value')


mnt_slider = widgets.IntSlider(
    value=128,
    min=10,
    max=512,
    step=1,
    description='new token:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',
)

input_textarea = widgets.Textarea(
    value='',
    description='Input:',
    layout=widgets.Layout(width='30%', height='250px'),
    disabled=False
)
output_textarea = widgets.Textarea(
    value='',
    description='Output:',
    layout=widgets.Layout(width='30%', height='250px'),
    disabled=False
)

submit_btn = widgets.Button(
    description="generate",
    disabled=False,
)


def generate(btn):
    input_text = input_textarea.value
    max_new_tokens = mnt_slider.value
    btn.disabled = True
    submit_btn.description = "Generating..."
    result = mt.generate(input_text, max_new_tokens=max_new_tokens)
    btn.disabled = False
    submit_btn.description = "generate"
    output_text = result[0]
    output_textarea.value = output_text


submit_btn.on_click(generate)

control_panel = widgets.HBox([mt_dropdown, setup_btn, precision_tbtn, device_tbtn])
talk_panel = widgets.HBox([input_textarea, widgets.VBox([mnt_slider, submit_btn]), output_textarea])
display(widgets.VBox([control_panel, talk_panel]))

# %%
mt_dropdown.index = 0
setup_btn.click()

# %% [markdown]
# LORA Tune MedQA

# %%
from peft import get_peft_model, LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

mt.model = get_peft_model(mt.model, peft_config)
mt.model.print_trainable_parameters()

# %%
bsz = 2

train_dst = MedQA('/nvme/guoyiqiu/coding/datasets/MedQA/data_clean/questions/US/train.jsonl', tokenizer=mt.tokenizer, size=100)
train_dl = DataLoader(train_dst, batch_size=bsz, shuffle=True, collate_fn=train_dst.collate_fn, num_workers=1)
test_dst = MedQA('/nvme/guoyiqiu/coding/datasets/MedQA/data_clean/questions/US/test.jsonl', tokenizer=mt.tokenizer, size=50)
test_dl = DataLoader(train_dst, batch_size=bsz, shuffle=False, collate_fn=test_dst.collate_fn, num_workers=1)


# %%
trainer_config = {
    "precision": "16-mixed",
    "accelerator": "auto",
    "devices": [1,4],
    "enable_checkpointing":True,
    "max_epochs":1,
    "strategy": "fsdp",
}

mt.clear_hook()
trainer = pl.Trainer(**trainer_config)
trainer.fit(mt, train_dl, test_dl)


