import torch

class TFConfig:
    from_pretrained = None
    pass

class TFModel:
    def __init__(self, config:TFConfig) -> None:
        self.model,self.tokenizer = self.gen_mt(config)
        
    def gen_mt(self, config):
        m = None
        t = None
        return m,t
    
    
    def predict(self,):
        pass
    
    def make_inputs(self, prompts, max_len=1024):
        token_lists = [self.tokenizer.encode(p) for p in prompts]
        for i in range(len(token_lists)):
            if len(token_lists[i]) >= max_len:
                token_lists[i] = token_lists[i][:max_len]
        maxlen = max(len(t) for t in token_lists)
        if "[PAD]" in self.tokenizer.all_special_tokens:
            pad_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index("[PAD]")]
        else:
            pad_id = 0
        input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
        # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
        attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
        return dict(
            input_ids=torch.tensor(input_ids).to(device),
            #    position_ids=torch.tensor(position_ids).to(device),
            attention_mask=torch.tensor(attention_mask).to(device),
        )
    def generate(self,*args, **kwargs):
        return self.model.generate(args, kwargs)
    
    def add_hook(self,)