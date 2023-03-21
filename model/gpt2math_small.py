from transformers import GPT2LMHeadModel, GPT2Tokenizer,GPT2TokenizerFast, GPT2Config
from tokenizers import Tokenizer
import torch.nn as nn
import json
import os

model_dir = "gpt2math/small"


class Gpt2mathSmall:
    def __init__(self):
        current_dir = os.getcwd()
        file_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(file_dir)
        json_config = json.load(open(f"{model_dir}/model_config.json"))
        config = GPT2Config.from_dict(json_config)
        self.model = GPT2LMHeadModel(config)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
        os.chdir(current_dir)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

