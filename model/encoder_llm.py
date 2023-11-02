from typing import Dict
from .llm import LLM
import torch.nn as nn

class DictEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    def forward(self, x : Dict):
        pass
        
        
class EncoderLLM(nn.Module):
    def __init__(self, mt_path, encoder):
        super(EncoderLLM, self).__init__()
        self.mt = LLM.from_pretrained(model_path=mt_path)
        self.encoder = encoder
        
    def forward(self, x):
        x = self.mt(x)
        x = self.encoder(x)
        return x