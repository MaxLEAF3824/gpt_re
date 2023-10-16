import torch
from typing import Dict, List, Union
from typing import Optional, Callable
from dataclasses import dataclass
import random
import hashlib
import requests
import json
import os
from treelib import Tree
import gpustat


def print_struct(data):
    def build_tree(data, tree, parent=None):
        for key, value in data.items() if isinstance(data, dict) else enumerate(data):
            if isinstance(value, list):
                node = tree.create_node(tag='list', parent=parent)
                build_tree(value, tree, parent=node.identifier)
            elif isinstance(value, tuple):
                node = tree.create_node(tag='tuple', parent=parent)
                build_tree(list(value), tree, parent=node.identifier)
            elif isinstance(value, dict):
                node = tree.create_node(tag='dict', parent=parent)
                build_tree(value, tree, parent=node.identifier)
            elif isinstance(value, torch.Tensor):
                node = tree.create_node(tag=f'torch.Tensor({list(value.shape)} device={value.device})', parent=parent)
            else:
                node = tree.create_node(tag=f'{type(value).__name__}', parent=parent)
            if isinstance(data, dict):
                node.tag = f'"{key}": {node.tag}'
        return tree

    tree = Tree()
    tree.create_node(tag='root', identifier=0)
    build_tree(data, tree, parent=0)
    tree.show()
    
def get_free_gpus():
    free_gpu_indexs = []
    gpus = gpustat.new_query()
    for gpu in gpus:
        if gpu.memory_used == 0 or (gpu.utilization <= 10 and gpu.memory_used < 5000):
            free_gpu_indexs.append(gpu.index)
    return free_gpu_indexs

class BaiduTrans:
    baidu_api = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    
    def __init__(self):
        self.appid = "123"
        self.key = "123"
        sk_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "trans_sk.json")
        if os.path.exists(sk_path):
            sk = json.load(open(sk_path))
            self.appid = sk['app_id']
            self.key = sk['secret_key']


    def generateSignature(self, query):
        salt = str(random.randint(0, 999999))
        string1 = self.appid + query + salt + self.key
        sign = hashlib.md5(string1.encode(encoding='UTF-8')).hexdigest()
        return salt, sign

    def query(self, q, lang_to='zh', lang_from="auto"):
        # print(f"query:{q}")
        salt, sign = self.generateSignature(q)
        req_data = {
            "q": q,
            "from": lang_from,
            "to": lang_to,
            "appid": self.appid,
            "salt": salt,
            "sign": sign
        }
        response = requests.post(self.baidu_api, data=req_data)
        # print(f"response:{response.json()}")
        result = ""
        for res in response.json()["trans_result"]:
            result = result + res['dst'] + '\n'
        return result[:-1]

class LeftPadding:
    """Context manager that change padding side to left."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ori_padding_side = self.tokenizer.padding_side
        
    def __enter__(self, *args, **kwargs):
        self.tokenizer.padding_side = 'left'

    def __exit__(self, *args, **kwargs):
        self.tokenizer.padding_side = self.ori_padding_side
    
class LoadWoInit:
    """Context manager that disable parameter initialization."""

    def __init__(self):
        self.constant_ = torch.nn.init.constant_
        self.zeros_ = torch.nn.init.zeros_
        self.ones_ = torch.nn.init.ones_
        self.uniform_ = torch.nn.init.uniform_
        self.normal_ = torch.nn.init.normal_
        self.kaiming_uniform_ = torch.nn.init.kaiming_uniform_
        self.kaiming_normal_ = torch.nn.init.kaiming_normal_

    def __enter__(self, *args, **kwargs):
        torch.nn.init.constant_ = lambda *args, **kwargs: None
        torch.nn.init.zeros_ = lambda *args, **kwargs: None
        torch.nn.init.ones_ = lambda *args, **kwargs: None
        torch.nn.init.uniform_ = lambda *args, **kwargs: None
        torch.nn.init.normal_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None

    def __exit__(self, *args, **kwargs):
        torch.nn.init.constant_ = self.constant_
        torch.nn.init.zeros_ = self.zeros_
        torch.nn.init.ones_ = self.ones_
        torch.nn.init.uniform_ = self.uniform_
        torch.nn.init.normal_ = self.normal_
        torch.nn.init.kaiming_uniform_ = self.kaiming_uniform_
        torch.nn.init.kaiming_normal_ = self.kaiming_normal_

def recursive_copy(x, float, clone, detach, device):
    if isinstance(x, torch.Tensor):
        x = x.to(device)
        x = x.detach() if detach else x
        x = x.clone() if clone else x
        x = x.float() if float else x
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v, float, clone, detach, device) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v, float, clone, detach, device) for v in x])
    elif x is None:
        return None
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."

@dataclass
class LLMHookConfig(Dict):
    """Config for LLMHook."""
    module_name: str
    layer: int = 0
    retain_output: bool = True
    output_save_func: Optional[Callable] = None
    retain_input: bool = False
    input_save_func: Optional[Callable] = None
    edit_output: Optional[Callable] = None
    float: bool = False
    clone: bool = False
    detach: bool = False
    device: str = "cpu"
    

class LLMHook:
    def __init__(self, module, config: LLMHookConfig):
        self.module = module
        self.config = config
        self.inputs = []
        self.outputs = []
        
        def hook(module, input, output):
            if config.retain_input:
                if config.input_save_func is not None:
                    in_save = config.input_save_func(module, input, output)
                else:
                    in_save = recursive_copy(input[0] if isinstance(input, tuple) else input,
                                             float=float, clone=config.clone, detach=config.detach, device=config.device)
                self.inputs.append(in_save)
            if config.retain_output:
                if config.output_save_func is not None:
                    out_save = config.output_save_func(module, input, output)
                else:
                    out_save = recursive_copy(output[0] if isinstance(output, tuple) else output,
                                              float=float, clone=config.clone, detach=config.detach, device=config.device)
                self.outputs.append(out_save)
            if config.edit_output:
                output = config.edit_output(module, input, output)
            return output

        self.hook = module.register_forward_hook(hook)

    def remove(self):
        self.inputs.clear()
        self.outputs.clear()
        self.hook.remove()

    def reset(self):
        self.inputs.clear()
        self.outputs.clear()


class LLMHooker:
    """Context manager that add multiple hooks to LLM."""
    
    def __init__(self, mt, config: Union[List[LLMHookConfig], LLMHookConfig]):
        self.mt = mt
        self.config_list = config if isinstance(config, list) else [config]
        self.hooks = []
    
    def __enter__(self):
        for config in self.config_list:
            module = getattr(self.mt, config.module_name)
            
            if not module:
                raise ValueError(f"Module {config.module_name} not found in LLM.")
            
            if isinstance(module, list):
                if config.layer < 0 or config.layer >= len(module):
                    raise ValueError(f"Module {config.module_name}[{config.layer}] out of range.")
                module = module[config.layer]
            
            if not isinstance(module, torch.nn.Module):
                raise ValueError(f"Module {config.module_name} is not a torch.nn.Module.")
            
            self.hooks.append(LLMHook(module, config))
        
        return self
    
    def __exit__(self, *args, **kwargs):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()