import torch
from typing import Union, List, Callable, Optional, Dict
from dataclasses import dataclass


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
class LLMHookerConfig(Dict):
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
    detach: bool = True
    device: str = "cpu"
    
class LLMHook:
    def __init__(self, module, config: LLMHookerConfig):
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
    
    def __init__(self, mt, config: Union[List[LLMHookerConfig], LLMHookerConfig]):
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