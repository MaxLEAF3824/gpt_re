attribute = "diastolic blood pressure"
lb = 30
llb = 0
ub = 50
uub = 100
unit = "mmHg"

size = 200
prefix_max_len = 5
suffix_max_len = 5
sample_args={
    "do_sample": True,
    "top_k": 100,
    "top_p": 10.0, 
    "temperature": 10.0
}
core_prompt = attribute + " is {} " + unit
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path="/home/sist/yangyuchen1/coding/vicuna-7b"
model = AutoModelForCausalLM.from_pretrained(path)
tok = AutoTokenizer.from_pretrained(path)
if not tok.eos_token:
    tok.add_special_tokens({
        "eos_token": "</s>",
    })
if not tok.bos_token:
    tok.add_special_tokens({
        "bos_token": "<s>",
    })
if not tok.unk_token:
    tok.add_special_tokens({
        "unk_token": "<unk>",
    })
if not tok.pad_token:
    tok.pad_token = tok.eos_token
tok.padding_side = 'left'
model.cuda()
prompts = pickle.load(open(f"prompts_diastolic_blood_pressure_200_vicuna_7b.pkl", "rb"))
target = f"truth: the normal range of diastolic blood pressure is between {lb} and {ub} mmHg"
target_len = len(tok(target).input_ids) - 1# remove bos token
inp = tok([(p.strip() + " " + target).strip() for p in prompts], return_tensors='pt', padding=True)
input_ids, attention_mask = inp['input_ids'], inp['attention_mask']
labels = torch.ones_like(input_ids) * -100
labels[:, -target_len:] = input_ids[:, -target_len:]

bsz = 4
test_ratio = 0.2
from torch.utils.data import DataLoader
dst = [[input_ids[i], attention_mask[i], labels[i]] for i in range(len(input_ids))]
train_dl = DataLoader(dst[:-int(test_ratio*size)], batch_size=bsz, num_workers=0)
test_dl = DataLoader(dst[-int(test_ratio*size):], batch_size=bsz, num_workers=0)
for d in train_dl:
    print(d)
    break

model.train()

epoch_num = 5
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
for epoch in range(epoch_num):
    for batch in train_dl:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()
        loss, logits, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
        print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
    for batch in test_dl:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()
        with torch.no_grad():
            loss, logits, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
        print(loss)
        acc = (logits.argmax(-1) == labels).float().mean()
        break
    
torch.save(model.state_dict(), f"ft_vicuna7b_{attribute}.pt")