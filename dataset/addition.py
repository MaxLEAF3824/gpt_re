import torch as t
import torch.utils.data as tud

class Addition(tud.Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        with open(data_path, 'r') as f:
            lines = f.readlines()
        self.data = [line.strip() for line in lines]
        self.input_ids, self.attention_mask, self.labels = self.make_inputs(self.tokenizer, self.data)

    def make_inputs(self, tokenizer, prompts):
        token_lists = [tokenizer.encode(p) for p in prompts]
        maxlen = max(len(t) for t in token_lists)
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
        eq = tokenizer.encode("=")[0]
        labels = []
        for tok in token_lists:
            eq_idx = tok.index(eq)
            labels.append([-100] * (maxlen - len(tok)) + [-100] * (eq_idx+1) + tok[eq_idx + 1:])
        input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
        attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
        return t.tensor(input_ids), t.tensor(attention_mask), t.tensor(labels)


    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)