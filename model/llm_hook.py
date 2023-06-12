from typing import Dict, List, Union
import torch

def recursive_copy(x, float=False, clone=False, detach=False, device=None):
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


class LLMHook(Dict):
    def __init__(self,
                 module,
                 name,
                 retain_output=True,
                 output_save_func=None,
                 retain_input=False,
                 input_save_func=None,
                 edit_output=None,
                 float=False,
                 clone=False,
                 detach=False,
                 device="cpu"):
        self.module = module
        self.name = name
        self.inputs = []
        self.outputs = []
        self.retain_output = retain_output
        self.output_save_func = output_save_func
        self.retain_input = retain_input
        self.input_save_func = input_save_func
        self.edit_output = edit_output
        self.float = float
        self.clone = clone
        self.detach = detach
        self.device = device

        def hook(module, input, output):
            if retain_input:
                if input_save_func is not None:
                    in_save = input_save_func(module, input, output)
                else:
                    in_save = recursive_copy(input[0] if isinstance(input, tuple) else input,
                                             float=float, clone=clone, detach=detach, device=device)
                self.inputs.append(in_save)
            if retain_output:
                if output_save_func is not None:
                    out_save = output_save_func(module, input, output)
                else:
                    out_save = recursive_copy(output[0] if isinstance(output, tuple) else output,
                                              float=float, clone=clone, detach=detach, device=device)
                self.outputs.append(out_save)
            if edit_output:
                output = edit_output(module, input, output)
            return output

        self.hook = module.register_forward_hook(hook)

    def remove(self):
        self.hook.remove()

