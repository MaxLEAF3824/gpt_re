import contextlib
from collections import OrderedDict
import torch


def get_module(model, num=None, kind=None):
    model_name = model._get_name()
    if kind not in [None, "attn", "mlp", "embed", "lm_head"]:
        raise LookupError(kind)
    if "GPT2" in model_name:
        if kind == "embed":
            name = "transformer.wte"
        elif kind == "lm_head":
            name = "lm_head"
        else:
            name = f'transformer.h.{num}{"" if kind is None else "." + kind}'
    else:
        raise LookupError(model)
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def get_hook_config(model, device, config="trace", output_fn=None):
    hook_config = []
    if "GPT2" in model._get_name():
        if config == "trace":
            hook_config.append({"module": get_module(model, kind="embed"),
                                "name": "embed",
                                "device": device})
            for l in range(model.config.n_layer):
                hook_config.append({"module": get_module(model, num=l, kind="attn"),
                                    "name": f"attn_{l}",
                                    "device": device})
                hook_config.append({"module": get_module(model, num=l, kind="mlp"),
                                    "name": f"mlp_{l}",
                                    "device": device})
                if l == 0:
                    hook_config.append({"module": get_module(model, num=l, kind=None),
                                        "name": f"block_{l}",
                                        "retain_input": True,
                                        "device": device})
                else:
                    hook_config.append({"module": get_module(model, num=l, kind=None),
                                        "name": f"block_{l}",
                                        "device": device})
            hook_config.append({"module": get_module(model, kind="lm_head"),
                                "name": "lm_head",
                                "device": device})
        else:
            raise LookupError(config)
    else:
        raise LookupError(model)
    return hook_config


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


class GPTHook(contextlib.AbstractContextManager):
    def __init__(self,
                 module,
                 name=None,
                 retain_output=True,
                 retain_input=False,
                 edit_output=None,
                 clone=False,
                 detach=False,
                 device="cpu"):
        self.name = name
        if name is None:
            name = module._get_name()
        self.input = None
        self.output = None

        def hook(module, input, output):
            if retain_input:
                self.input = recursive_copy(input[0] if len(input) == 1 else input,
                                            clone=clone, detach=detach, retain_grad=False, device=device)
            if retain_output:
                self.output = recursive_copy(output, clone=clone, detach=detach, device=device)
            if edit_output:
                output = edit_output(module, input, output)
            return output

        self.hook = module.register_forward_hook(hook)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.hook.remove()


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    def __init__(self,
                 model,
                 hook_config=None,
                 device="cpu"):
        if hook_config is None:
            hook_config = get_hook_config(model, device)
        else:
            for m in hook_config:
                m["device"] = device
        for i, m in enumerate(hook_config):
            self[m.get("name", i)] = GPTHook(**m)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        for name,hook in self.items():
            hook.__exit__(type, value, traceback)
