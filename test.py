import wandb
import os
import random
import torch
import json
from model import *
from sklearn.decomposition import PCA
import numpy as np
    

def calcu_score(neuron_act, reading_vectors, mean_vector, std_vector):
    """
    reading_vectors: [layer, hidden_size]
    mean_vector: [layer * hidden_size]
    std_vector: [layer * hidden_size]
    """
    neuron_act = neuron_act.transpose(0,1) # [seq_len, layer, hidden_size]
    seq_len, layer, hidden_size = neuron_act.shape
    neuron_act = neuron_act.reshape(seq_len, layer * hidden_size) # [seq_len, layer * hidden_size]
    neuron_act = (neuron_act - mean_vector) / std_vector
    neuron_act = neuron_act.reshape(seq_len, layer, hidden_size) # [seq_len, layer, hidden_size]
    scores = (neuron_act * reading_vectors.repeat(seq_len, 1, 1)).sum(-1) # [seq_len, layer]
    return scores.mean()


def compute_reading_vectors(neuron_acts):
    diff = []
    for (act1,act2) in neuron_acts:
        l = min(act1.shape[1],act2.shape[1])
        diff.append(act1[:,:l,:] - act2[:,:l,:])
    diff = torch.cat(diff, dim=1).transpose(0,1) # [sum(seq_len), layer, hidden_size]
    sample_size, n_layer, hidden_size = diff.shape
    diff = diff.reshape(sample_size, -1).numpy() # [sample_size, layer * hidden_size]
    mean_vector = np.mean(diff, axis=0)
    std_vector = np.std(diff, axis=0)
    diff = (diff - mean_vector) / std_vector
    pca = PCA(n_components=1)
    pca.fit(diff)
    reading_vectors = pca.components_[0].reshape(n_layer, hidden_size) # [layer, hidden_size]
    # print('reading_vectors: ', reading_vectors.shape)
    reading_vectors = torch.from_numpy(reading_vectors)
    mean_vector = torch.from_numpy(mean_vector)
    std_vector = torch.from_numpy(std_vector)
    return reading_vectors, mean_vector, std_vector, pca.explained_variance_ratio_[0]


def collect_neuron_acts(mt, dst, capture_window, layers, local_bsz=32):
    data_bsz = local_bsz // 2
    dst = [dst[i:i+data_bsz] for i in range(0, len(dst), data_bsz)]
    neuron_acts = []
    for batch_pairs in tqdm(dst):
        pairs = []
        for pair in batch_pairs:
            pairs += pair
        prompt_lens = [len(mt.tok(s['input'])['input_ids']) for s in pairs]
        seq_lens = [len(mt.tok(s['input']+s['output'])['input_ids']) for s in pairs]
        with PaddingSide(mt.tok, 'right'):
            input_ids = mt.tok([s['input']+s['output'] for s in pairs], return_tensors='pt', padding=True)['input_ids']
        hook_configs = [LLMHookerConfig(module_name='block', layer=l) for l in layers]
        with torch.no_grad(), LLMHooker(mt, hook_configs) as hooker:
            mt.model(input_ids=input_ids.to(mt.model.device))
            sentences_repr = torch.stack([h.outputs[0] for h in hooker.hooks]).transpose(0,1) # [bsz, layer, seq_len, hidden_size]
        batch_neuron_acts = []
        for i,repr in enumerate(sentences_repr):
            prompt_len = prompt_lens[i]
            seq_len = seq_lens[i]
            start = prompt_len + capture_window[0] if capture_window[0] >= 0 else seq_len + capture_window[0]
            end = prompt_len + capture_window[1] if capture_window[1] > 0 else seq_len + capture_window[1]
            batch_neuron_acts.append(repr[:,start:end,:])
        batch_neuron_acts = [[batch_neuron_acts[i],batch_neuron_acts[i+1]] for i in range(0, len(batch_neuron_acts), 2)]
        neuron_acts.extend(batch_neuron_acts)
    return neuron_acts # [layer, window_size, hidden_size]


def full_pipeline(mt, train_dst, test_dst, capture_window, compare_window, layers=None, local_bsz=64):
    if layers is None:
        layers = list(range(mt.n_layer))
    neuron_acts = collect_neuron_acts(mt, train_dst, capture_window, layers, local_bsz=local_bsz)
    rv, mv, sv, importance = compute_reading_vectors(neuron_acts)
    test_neuron_acts = collect_neuron_acts(mt, test_dst, compare_window, layers, local_bsz=local_bsz)
    scores = [[calcu_score(tna, rv, mv, sv),calcu_score(fna, rv, mv, sv)] for (tna, fna) in test_neuron_acts]
    mean_diff = np.mean([s[0]-s[1] for s in scores])
    acc = sum([1 for s in scores if s[0]>s[1]])/len(scores)
    return acc, mean_diff, importance


random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# os.environ["WANDB_NOTEBOOK_NAME"] = "re.ipynb"

mt = LLM.from_pretrained(model_path="/home/cs/yangyuchen/guoyiqiu/my_models/internlm-chat-20b").cuda(get_free_gpus()[0])

tf_dst = json.load(open("data/true_false_dataset.json"))
# IID_Hard
reserved_topic = "Medical"
train_dst_iid_hard = [[td,fd] for (td,fd) in tf_dst if td['topic'] != "Medical"]
test_dst_iid_hard = [[td,fd] for (td,fd) in tf_dst if td['topic'] == "Medical"]
# IID_Weak
random.shuffle(tf_dst)
train_dst_iid_weak = tf_dst[:len(train_dst_iid_hard)]
test_dst_iid_weak = tf_dst[len(train_dst_iid_hard):]
# OOD
reserved_topic = "Medical"
prompt = "USER:Tell me a fact.\nAssistant:"
train_dst_ood = [[td,fd] for (td,fd) in tf_dst if td['topic'] != "Medical"]
test_dst_ood = [[dict(input=prompt,output=td['output'],topic=td['topic'],label=True),
                 dict(input=prompt,output=fd['output'],topic=fd['topic'],label=False)] 
                for (td,fd) in tf_dst if td['topic'] == "Medical"]

config = {
    "capture_window": (0,0),
    "compare_window": (0,0),
    "local_bsz": 32,
}

wandb.init(config=config, 
           project="lat layer sweep", 
           name="internlm_20b_seed42",
           dir="output/lat_layer_sweep",
           job_type="inference")


for l in tqdm(mt.n_layer):
    layers = [l]
    acc, mean_diff, importance = full_pipeline(mt=mt, train_dst=train_dst_iid_weak, test_dst=test_dst_iid_weak,layers=layers, **config)
    # print(f"iid weak\nimportance: {importance:.4f}\nacc: {acc:.4f},\nmean_diff: {mean_diff:.4f}")
    wandb.log({"iid weak": acc if acc>0.5 else 1-acc, "mean_diff": abs(mean_diff), "importance": importance}, step=l)
    acc, mean_diff, importance = full_pipeline(mt=mt, train_dst=train_dst_iid_hard, test_dst=test_dst_iid_hard,layers=layers, **config)
    # print(f"iid hard\nimportance: {importance:.4f}\nacc: {acc:.4f},\nmean_diff: {mean_diff:.4f}")
    wandb.log({"iid hard": acc if acc>0.5 else 1-acc, "mean_diff": abs(mean_diff), "importance": importance}, step=l)
    acc, mean_diff, importance = full_pipeline(mt=mt, train_dst=train_dst_ood, test_dst=test_dst_ood, layers=layers, **config)
    # print(f"ood\nimportance: {importance:.4f}\nacc: {acc:.4f},\nmean_diff: {mean_diff:.4f}")
    wandb.log({"ood": acc if acc>0.5 else 1-acc, "mean_diff": abs(mean_diff), "importance": importance}, step=l)

wandb.finish()