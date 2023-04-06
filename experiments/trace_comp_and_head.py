import os 
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
import torch
import torch.nn.functional as F
from dataset.knowns import KnownsDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from gpthook import TraceDict, get_hook_config
from viz_tool import *

import pickle
from datasets import load_dataset
import pdb
from tqdm import tqdm
from collections import defaultdict
from dataset.utils import *


def make_inputs(tokenizer, prompts, max_len=1024, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    for i in range(len(token_lists)):
        if len(token_lists[i]) >= max_len:
            token_lists[i] = token_lists[i][:max_len]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
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
 

def attn_custom(attn, xv, attn_weight=None, xq=None, xk=None, head_mask=None):
    if xq is not None and xk is not None:
        query = attn.c_attn(xq).split(attn.split_size, dim=2)[0]
        key = attn.c_attn(xk).split(attn.split_size, dim=2)[1]
        query = attn._split_heads(query, attn.num_heads, attn.head_dim)
        key = attn._split_heads(key, attn.num_heads, attn.head_dim)
        attn_weight = torch.matmul(query, key.transpose(-1, -2))
        attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1) # [bsz, head, seq, seq]
    elif attn_weight is None:
        raise ValueError("attn_weight or (xq, xk) should be provided")
    if head_mask:
        attn_weight[:, head_mask] = 0
    value = attn.c_attn(xv).split(attn.split_size, dim=2)[2]
    value = attn._split_heads(value, attn.num_heads, attn.head_dim)
    attn_output = torch.matmul(attn_weight, value)
    attn_output = attn._merge_heads(attn_output, attn.num_heads, attn.head_dim)
    attn_output = attn.c_proj(attn_output)
    # attn_output = attn.resid_dropout(attn_output)
    return attn_output, attn_weight


def calculate_effect(model, x_fixed_fn_list, tds, save_dir="output"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = f"{model._get_name()}_{model.config.n_layer}"
    with open(tds, "rb") as f:
        tds = pickle.load(f)
    print("tds ready")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.to(device)
    test_num = (len(tds) - 1) * tds[0]['block_0'].input.shape[0] + tds[-1]['block_0'].input.shape[0]
    n_layer = model.config.n_layer

    for x_fixed_fn in x_fixed_fn_list:
        fn_name = x_fixed_fn.__name__
        x_fixed_cos_sims = np.zeros(n_layer)
        prob_diffs = []
        ces = []
        acc = 0
        print(f"calculating {fn_name}")
        for td in tqdm(tds):
            x_fixed = x_fixed_fn(model, td)

            # get x0 and xl for the last token
            x_fixed = x_fixed[:, -1, :].unsqueeze(1)  # [bsz, 1, hidden]
            xi = [model.transformer.ln_f(td[f"block_{l}"].output[0].to(device))
                  for l in range(n_layer)]  # x_1, ..., x_L List[[bsz, seq, hidden]]
            xl = torch.stack(xi, dim=0).transpose(0, 1)[:, :, -1, :]  # x1, ..., xL [bsz, layer, hidden]

            # calculate the cosine similarity
            x_fixed_cos_sims += torch.cosine_similarity(x_fixed, xl,
                                                        dim=-1).sum(dim=0).cpu().detach().numpy()  # [bsz, layer]

            # calculhate the probability difference
            logits = model.lm_head(xl)[:, -1, :]
            prob = torch.softmax(logits, dim=-1)
            logits_fixed = model.lm_head(x_fixed)[:, -1, :]
            prob_fixed = torch.softmax(logits_fixed, dim=-1)
            for i in range(prob.shape[0]):
                diff = torch.max(prob[i, :]).item() - prob_fixed[i, torch.argmax(prob[i, :])].item()
                prob_diffs.append(diff)
            acc += (torch.argmax(prob, dim=-1) == torch.argmax(prob_fixed, dim=-1)).sum().item()

            ces.append(torch.nn.functional.cross_entropy(logits, logits_fixed).item())

        plot_bar(f"Cos Sim of xi and x_fixed {fn_name}", x_fixed_cos_sims / test_num, f"{save_dir}/{model_name}_{fn_name}_cos_sim.png")

        plot_hist(f"Prob Diff with func {fn_name}", prob_diffs, f"{save_dir}/{model_name}_{fn_name}_prob_diff.png")

        plot_hist(f"CE Loss with func {fn_name}", ces, f"{save_dir}/{model_name}_{fn_name}_ce_loss")

        print(f"acc:{round(acc/test_num, 4)} "
              f"cos_sim:{round(x_fixed_cos_sims[-1]/test_num, 4)} "
              f"prob_diff:{round(np.mean(prob_diffs), 4)} "
              f"ce loss:{round(np.mean(ces), 4)} ")


def without_v(model, td):
    # calculate the x_fixed without virtual terms
    x0 = td["block_0"].input.to(model.device)  # x_0 [bsz, seq, hidden]
    x_fixed = x0
    for i in range(model.config.n_layer):
        block = model.transformer.h[i]
        attn_weight = td[f'attn_{i}'].output[2].to(model.device)
        attn_output, _ = attn_custom(block.attn, xv=block.ln_1(x0), attn_weight=attn_weight)
        mlp_output = td[f'mlp_{i}'].output.to(model.device)
        x_fixed += (attn_output + mlp_output)  # [bsz, seq, hidden]
    x_fixed = model.transformer.ln_f(x_fixed)
    return x_fixed


def without_a(model, td):
    x0 = td["block_0"].input.to(model.device)  # x_0 [bsz, seq, hidden]
    x_fixed = td['block_0'].output[0].to(model.device)
    x_in = td["block_0"].output[0].to(model.device)
    pre_attn = td["attn_0"].output[0].to(model.device)
    for i in range(1, model.config.n_layer):
        block = model.transformer.h[i]
        x_in = x_in - pre_attn
        attn_output,_ = attn_custom(block.attn, xv=block.ln_1(x0), xq=block.ln_1(x_in), xk=block.ln_1(x_in))
        mlp_output = td[f'mlp_{i}'].output.to(model.device)
        x_fixed += (attn_output + mlp_output)  # [bsz, seq, hidden]
        x_in = x_fixed
        pre_attn = attn_output
    x_fixed = model.transformer.ln_f(x_fixed)
    return x_fixed


def without_m(model, td):
    x0 = td["block_0"].input.to(model.device)  # x_0 [bsz, seq, hidden]
    x_fixed = td['block_0'].output[0].to(model.device)
    x_in = td["block_0"].output[0].to(model.device)
    pre_mlp = td["mlp_0"].output[0].to(model.device)
    for i in range(1, model.config.n_layer):
        block = model.transformer.h[i]
        x_in = x_in - pre_mlp
        attn_output,_ = attn_custom(block.attn, xv=block.ln_1(x0), xq=block.ln_1(x_in), xk=block.ln_1(x_in))
        mlp_output = td[f'mlp_{i}'].output.to(model.device)
        x_fixed += (attn_output + mlp_output)  # [bsz, seq, hidden]
        x_in = x_fixed
        pre_mlp = mlp_output
    x_fixed = model.transformer.ln_f(x_fixed)
    return x_fixed


def collect_trace_dicts(model, tokenizer, input_texts, device="cuda", max_len=200, save_dir=None):
    tds = []

    model.eval()
    model.to(device)
    for input_text in tqdm(input_texts):
        td, _ = get_clean_td(model, tokenizer, input_text, device=device, save_device="cpu", max_len=max_len)
        tds.append(dict(td))

    model_name = f"{model._get_name()}_{model.config.n_layer}"
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pickle.dump(tds, open(f"{save_dir}/tds_{model_name}_{len(input_texts)}.pkl", "wb"))

    return tds


def get_clean_td(model, tokenizer, input_text, device="cuda", save_device="cpu", max_len=200):
    max_len = min(max_len, model.config.n_ctx - 1)
    inp = make_inputs(tokenizer, [input_text], max_len=max_len, device=device)

    with torch.no_grad(), TraceDict(model, device=save_device) as td:
        logits = model(**inp, output_attentions=True)["logits"]

    return td, logits


def get_x0(model, inp):
    x0 = model.transformer.wte(inp["input_ids"])
    input_shape = inp["input_ids"].size()
    position_ids = torch.arange(0, input_shape[-1] + 0, dtype=torch.long, device=model.device)
    position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    position_embeds = model.transformer.wpe(position_ids)
    x0 = x0 + position_embeds
    return x0


def trace_comp_patch(model, inp, x0,
                     layer: int, t_idxs: list,
                     comp, comp_kind,
                     output_attentions=False):
    if layer == 0:
        raise ValueError("you cant trace layer 0")

    # x0 = get_x0(model, inp)
    ai = None
    attn_weight = {}
    def output_fn(module, input, output):
        '''将x_i的指定位置减去comp项, 生成新的xq,xk'''
        xi = input[0]
        xq = xi
        xk = xi
        xv = x0
        # mask = torch.rand_like(comp[:, t_idxs, :])
        if comp_kind == "key":
            xk[:, t_idxs, :] = xk[:, t_idxs, :] - comp[:, t_idxs, :]
        elif comp_kind == "query":
            xq[:, t_idxs, :] = xq[:, t_idxs, :] - comp[:, t_idxs, :]
        else:
            raise ValueError("unseen composition kind")
        ai, _ = attn_custom(module.attn, xv=module.ln_1(xv), xq=module.ln_1(xq), xk=module.ln_1(xk))
        attn_weight[0] = _
        mi = module.mlp(module.ln_2(ai+xi))
        hidden_states = xi + ai + mi
        if output_attentions:
            return hidden_states, output[1], output[2]
        return hidden_states, output[1]

    hook_config = [{"module": model.transformer.h[layer],
                    "name": f"block_{layer}",
                    "retain_output":False,
                    "edit_output": output_fn}]
    if output_attentions:
        hook_config.append({"module": model.transformer.h[layer].attn,
                            "name": f"attn_{layer}"})
    with torch.no_grad(), TraceDict(model, hook_config, device="cuda") as td:
        logits = model(**inp, output_attentions=output_attentions)["logits"]

    prob = F.softmax(logits, dim=-1)
    # pdb.set_trace()
    
    if output_attentions:
        td[f'attn_{layer}'].output = ((td[f'attn_{layer}'].output[0], ai), td[f'attn_{layer}'].output[1], 
                               (td[f'attn_{layer}'].output[2], attn_weight[0]))
    
    return prob, td


def calculate_comp_flow(model, tokenizer, input_text, comp_key, comp_kind):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    inp = make_inputs(tokenizer, [input_text], device=device)
    input_tokens = [tokenizer.decode([t]) for t in inp['input_ids'][0]]
    # get clean td
    with torch.no_grad(), TraceDict(model, device="cpu") as clean_td:
        logits = model(**inp, output_attentions=True)['logits'] # [bsz, seq, vocab]
    clean_prob = F.softmax(logits, dim=-1)
    gt_idx = torch.argmax(clean_prob[:,-1,:], dim=-1)
    answer = tokenizer.decode(gt_idx)
    gt_prob = clean_prob[:,-1,gt_idx]
    
    x0 = clean_td["block_0"].input.to(model.device)
    table = []
    attn_weight_diff = []
    for layer in range(1, model.config.n_layer):
        if comp_key == "attn":
            comp = clean_td[f"{comp_key}_{layer - 1}"].output[0] 
        else:
            comp = clean_td[f"{comp_key}_{layer - 1}"].output
        comp = comp.to(device)
        column = []
        for t_idx in range(len(inp['input_ids'][0])):
            prob, td = trace_comp_patch(model, inp, x0, layer, [t_idx], comp, comp_kind)
            column.append(gt_prob - prob[:,-1,gt_idx])
        column = torch.vstack(column)
        table.append(column)
        # corrupt all tokens
        t_idxs = list(range(len(inp['input_ids'][0])))
        prob, td = trace_comp_patch(model, inp, x0, layer, t_idxs, comp, comp_kind, output_attentions=True)
        attn_weight_o, attn_weight_fixed = td[f'attn_{layer}'].output[2]
        device2 = attn_weight_fixed.device
        # pdb.set_trace()
        attn_weight_diff.append((attn_weight_o-attn_weight_fixed).abs().sum(dim=-1).sum(dim=-1))
    attn_weight_diff = torch.vstack(attn_weight_diff)
    table = torch.stack(table).squeeze()
    return {"table":table.transpose(0,1).cpu(),
            "comp_key":comp_key,
            "comp_kind":comp_kind,
            "input_tokens": input_tokens,  
            "answer":answer,
            "attn_weight_diff":attn_weight_diff.cpu()}

def calculate_head_flow(model, tokenizer, input_text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    model.to(device)
    model.eval()
    inp = make_inputs(tokenizer, [input_text], device=device)
    logits = model(**inp)['logits'] # [bsz, seq, vocab]
    clean_prob = F.softmax(logits[:,-2,:], dim=-1)
    gt_idx = torch.argmax(clean_prob, dim=-1)
    answer = tokenizer.decode(gt_idx)
    gt_prob = clean_prob[:, gt_idx]
    
    # x0 = clean_td["block_0"].input.to(model.device)
    table = []
    for layer in range(model.config.n_layer):
        column = []
        for h_idx in range(model.config.n_head):
            # print(f"layer {layer}, head {h_idx}")
            head_mask = torch.ones(model.config.n_layer, model.config.n_head, device=device)
            head_mask[layer, h_idx] = 0
            logits = model(**inp, head_mask=head_mask)["logits"]
            prob = F.softmax(logits[:,-2,:], dim=-1)
            column.append((gt_prob - prob[:, gt_idx]).detach().cpu())
        column = torch.vstack(column)
        table.append(column)
    table = torch.stack(table).squeeze()
    return {"table":table.transpose(0,1),
            "input_tokens": list(range(model.config.n_head)),
            "answer":answer}


def plot_trace_result(result, save: str = None, title=None, xlabel=None, modelname=None):
    differences = result["table"]
    low_score = differences.min()
    answer = result["answer"]
    comp_key = result.get("comp_key","head")
    comp_kind = result.get("comp_kind","head")
    kind = comp_key+"_"+comp_kind
    attn_weight_diff = result.get("attn_weight_diff", None)
    labels = list(result["input_tokens"])

    with plt.rc_context(rc={"font.family": "DejaVu Sans"}):
        # fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        fig, ax = plt.subplots()
        h = ax.pcolor(
            differences,
            cmap={"attn_key": "Blues", "attn_query": "Purples", "mlp_key": "Greens", "mlp_query": "Reds"}
            .get(kind, "Blues"),
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(differences.shape[1])])
        ax.set_xticklabels(list(range(differences.shape[1])))
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT2"
        ax.set_title(f"Impact of Corrupt {comp_key} in Softmax {comp_kind} Side")
        if title:
            ax.set_title(title)
        ax.set_xlabel(f"layer within {modelname}")
        cb = plt.colorbar(h)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        
        if save:
            save_dir, file_name = os.path.split(save)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
        if attn_weight_diff is not None:
            plot_matrix("Attn Weight Diff", attn_weight_diff, xlabel="layer",ylabel="head", save=f"{save}_ADiff.png")


if __name__ == "__main__":
    model_dir = "/mnt/petrelfs/guoyiqiu/coding/huggingface_models/"
    model_dir = ""
    data_dir = "/mnt/petrelfs/guoyiqiu/coding/data"
    data_dir = ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "gpt2-xl"
    model = AutoModelForCausalLM.from_pretrained(f"{model_dir}{model_name}")
    model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}{model_name}")
    dst = KnownsDataset(data_dir, size=100)
    # input_text = "Context: Human die. Tom is human. Judugement: Tom would die. Is the judgement correct? The answer is yes"

    results = []
    for i, data in tqdm(enumerate(dst), total=len(dst)):
        # print(i)
        input_text = data['prompt']
        next_t_idx = tokenizer.encode(data['prediction'])[0]
        # answer = tokenizer.convert_ids_to_tokens(next_t_idx)
        answer = tokenizer.decode(next_t_idx)
        input_text += answer
        result = calculate_head_flow(model, tokenizer, input_text)
        # plot_trace_result(result, title=f"Impact of Corrupt Head", xlabel=input_text, save=f"output/{model_name}_head_heatmap/{i}.png")
        # for comp_key, comp_kind in [("attn","key"),("mlp","key")]:
            # result = calculate_comp_flow(model, tokenizer, input_text, comp_key, comp_kind)
            # plot_trace_result(result, title=f"{i}", save="output/{model_name}_comp_heatmap/{i}.png")
        results.append(result)
    import pickle 
    pickle.dump(results, open("output/{model_name}_head_heatmap/results.pkl", "wb"))
