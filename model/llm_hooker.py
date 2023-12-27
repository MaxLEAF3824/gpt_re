import torch
from typing import Union, List, Callable, Optional, Dict
from dataclasses import dataclass


def default_input_save_func(module, input_args, input_kwargs, output):
    if isinstance(input_args, tuple):
        return input_args[0]
    return input_args

def default_output_save_func(module, input_args, input_kwargs, output):
    if isinstance(output, tuple):
        return output[0]
    return output

def recursive_copy(x, float, clone, detach, device):
    if isinstance(x, torch.Tensor):
        x = x.to(device) if device else x
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
        raise ValueError(f"Unknown type {type(x)} cannot be broken into tensors.")

@dataclass
class LLMHookerConfig(Dict):
    """Config for LLMHooker."""
    module_name: str
    layer: int = 0
    save_output: Union[bool,Callable] = True
    save_input: Union[bool,Callable] = False
    edit_output: Optional[Callable] = None
    float: bool = False
    clone: bool = False
    detach: bool = True
    device: str = "cpu"
    
class LLMHook:
    def __init__(self, module, config: LLMHookerConfig):
        self.module = module
        self.config = config
        self.inputs = []
        self.outputs = []
        
        def hook(module, input_args, input_kwargs, output):
            if config.save_input:
                if isinstance(config.save_input, bool):
                    config.save_input = default_input_save_func
                in_save = config.save_input(module, input_args, input_kwargs, output)
                in_save = recursive_copy(in_save, float=float, clone=config.clone, detach=config.detach, device=config.device)
                self.inputs.append(in_save)
            if config.save_output:
                if isinstance(config.save_output, bool):
                    config.save_output = default_output_save_func
                out_save = config.save_output(module, input_args, input_kwargs, output)
                out_save = recursive_copy(out_save, float=float, clone=config.clone, detach=config.detach, device=config.device)
                self.outputs.append(out_save)
            if config.edit_output:
                output = config.edit_output(module, input_args, input_kwargs, output)
            return output

        self.hook = module.register_forward_hook(hook, with_kwargs=True)

    def remove(self):
        self.inputs.clear()
        self.outputs.clear()
        self.hook.remove()

    def reset(self):
        self.inputs.clear()
        self.outputs.clear()

class LLMHooker:
    """Context manager that add multiple hooks to LLM."""
    
    def __init__(self, mt, config: Union[List[LLMHookerConfig], LLMHookerConfig]):
        self.mt = mt
        self.configs = config if isinstance(config, list) else [config]
        self.hooks = []
    
    def __enter__(self):
        for config in self.configs:
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