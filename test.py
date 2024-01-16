# 生成v3.3/checkout_data_train训练数据集
# 数据来源：各大内科，使用bert_score筛选与化验相关度高的数据，进一步清洗了出院诊断，并对出院诊断做了归一化
# 进一步增加了stage1 Encoder预训练数据，任务：还原化验单的异常表项
import json
import os
import pandas as pd
from transformers import AutoTokenizer, BertModel
from tqdm import tqdm
import re
import random
import multiprocessing
import torch
import faiss


data_path = os.path.join(os.environ['my_datasets_dir'], "ninth/checkout_data_im_with_checks.json")
ft_eval_ratio = 0.05
pt_eval_ratio = 0.01
bert_score_f1_threshold = 0.8
max_dict_token_num = 4096  # max_length for dict token num
max_pt_token_num = 80  # max length for pretrain_input text + pt_output text
max_ft_token_num = 768  # max length for ft_input text + ft_output text
max_ft_text_token_num = 8192  # max length for ft_input text + text dicts + ft_output text
max_ft_data_size = 20000  # FT 训练数据集规模
max_pt_data_size = 20000  # Pretrain 训练数据集规模
max_num_abnormal_sample = 10  # 每个化验单最多sample的异常检验项目数
seed = 42
version = "3.3"
table_token = '[TABLE]'
bios_index = faiss.read_index(os.path.join(os.environ['my_datasets_dir'], "bios_v2.2_release/CoreData/TermDiseaseZHEmbedding_HNSW64.index"))
bios_term2cid = json.load(open((os.path.join(os.environ['my_datasets_dir'], "bios_v2.2_release/CoreData/TermDiseaseZH.json"))))
cid2bios_term = {v:k for (k,v) in bios_term2cid.items()}
bios_terms = list(bios_term2cid.keys())
bert = BertModel.from_pretrained(os.path.join(os.environ['my_models_dir'], 'bert-base-chinese'))
bert.eval()
term2bterm = json.load(open(os.path.join(os.environ['my_datasets_dir'], "bios_v2.2_release/CoreData/term2bterm.json")))

pt_templates = [
    f"{table_token} 请根据化验单结果输出病人可能患有的疾病。输出:",
    f"{table_token} 请依据化验单的结果判断病人可能的疾病。输出:",
    f"{table_token} 根据化验单结果，请推测病人可能罹患的疾病。输出:",
    f"{table_token} 请分析化验单结果并指出病人可能遭受的疾病。输出:",
    f"{table_token} 基于化验单结果，预测病人可能患有哪些疾病。",
    f"{table_token} 请从化验单结果中判断出病人可能的疾病。输出:",
    f"{table_token} 根据化验单的结果，推断病人可能的健康问题。输出:",
    f"{table_token} 请查看化验单结果并识别可能的疾病。输出:",
    f"{table_token} 请解读化验单结果，判断病人可能面临的疾病。输出:",
    f"{table_token} 根据化验单结果，确定病人可能患的疾病。输出:",
    f"{table_token} 分析化验单结果并预判病人可能的疾病。输出:"
]

ft_text_template = "性别:{}\n年龄:{}\n入院时主要症状及体征:{}\n特殊检查及重要会诊:{}\n出院诊断:"
ft_template = "化验信息:{}文字信息:{}"
llm_tok = AutoTokenizer.from_pretrained(os.path.join(os.environ['my_models_dir'], 'internlm-7b'), trust_remote_code=True)
bert_tok = AutoTokenizer.from_pretrained(os.path.join(os.environ['my_models_dir'], 'bert-base-chinese'), trust_remote_code=True)
bert_tok.add_tokens([table_token])
random.seed(seed)
tqdm.pandas()

# get data_df
data = json.load(open(data_path))
data = random.sample(data, len(data))
print(f'full df size: {len(data)}')

def topk_index(index: pd.Series, k):
    count = 0
    for i in index.index:
        if index[i]:
            count += 1
            if count > k:
                index[i] = False
    return index

def normalize_term(term):
    if term in term2bterm:
        return term2bterm[term]
    with torch.no_grad():
        term_embedding = bert(bert_tok(term, return_tensors='pt')['input_ids'])['last_hidden_state'][:,0].cpu().numpy()
    D,I = bios_index.search(term_embedding, 1)
    cid = bios_term2cid[bios_terms[I[0][0]]]
    bios_term = cid2bios_term[cid]
    term2bterm[term] = bios_term
    return bios_term
    
def get_abnormal_items(dict):
    abnormal_items = []
    for d in dict:
        for (k, v) in d['data'].items():
            if v in ["空", "阳性", "阳性(+)", "弱阳性", "极弱阳性"]:
                abnormal_items.append(k)
    if abnormal_items:
        num_sample = min(len(abnormal_items), max_num_abnormal_sample)
        abnormal_items = random.sample(abnormal_items, num_sample)
        return "，".join(abnormal_items)
    else:
        return "无"

def get_label_data_str(dict):
    dict_str = ""
    for d in dict:
        for (k, v) in d['data'].items():
            if v == "N":
                v = "正常"
            elif v == "空":
                v = "异常"
            dict_str += f"{k}{v} "
        dict_str += "[SEP]"
    return dict_str

def preprocess_data(d):    
    label_data = [[dict(header=d['header'], data=d['data']) for d in d['完整化验结果']]]
    raw_data = [[dict(header=d['header'], data=d['raw_data']) for d in d['完整化验结果']]]
    label_data_str = get_label_data_str(d['完整化验结果'])
    
    diagnosis = d['出院诊断']
    diagnosis = diagnosis.replace("「", "").replace("」", "").replace("\n", "").replace(" ", "").replace("(", "（").replace(")", "）")
    diagnosis = re.sub("（[^）]*）", "", diagnosis)
    ft_output = "，".join([term2bterm.get(t.strip(), t.strip()) for t in diagnosis.split("，")])
    ft_input = ft_text_template.format(d['性别'], d['年龄'], d['入院时主要症状及体征'], d['特殊检查及重要会诊'])
    ft_input = ft_input.replace("「", "").replace("」", "")
    for o in ft_output.split("，"):
        ft_input = ft_input.replace(o, "")
    ft_text_dict_input = ft_template.format(label_data_str, ft_input)
    
    pt_input = random.choice(pt_templates)
    pt_output = ft_output
    
    num_dict_token = len(bert_tok(label_data_str, add_special_tokens=False)['input_ids'])
    num_ft_text_dict_token = len(llm_tok(ft_text_dict_input, add_special_tokens=False)['input_ids'])
    num_pt_token = len(llm_tok(pt_input+pt_output)['input_ids'])
    num_ft_token = len(llm_tok(ft_input+ft_output)['input_ids'])
    
    d['pt_input'] = pt_input
    d['pt_output'] = pt_output
    d['label_data'] = label_data
    d['raw_data'] = raw_data
    d['label_data_str'] = label_data_str
    d['ft_input'] = ft_input
    d['ft_text_dict_input'] = ft_text_dict_input
    d['ft_output'] = ft_output
    d['num_dict_token'] = num_dict_token
    d['num_ft_text_dict_token'] = num_ft_text_dict_token
    d['num_pt_token'] = num_pt_token
    d['num_ft_token'] = num_ft_token
    
    return d

print(f"generating data_df")
preprocessed_data = []

process_num = 8
pool = multiprocessing.Pool(process_num)
print(f"using {process_num} process")
for output in tqdm(pool.imap(preprocess_data, data), total=len(data)):  
	preprocessed_data.append(output)
pool.close()

# for d in tqdm(data):
#     preprocessed_data.append(preprocess_data(d))

data_df = pd.DataFrame(preprocessed_data)

# filter and organize data_df
ft_index = (data_df['门诊诊断'].apply(len) > 0) \
    & (data_df['出院诊断'].apply(len) > 0) \
    & (data_df['门诊出院bert_score_f1'] < bert_score_f1_threshold) \
    & (data_df['num_ft_token'] < max_ft_token_num) \
    & (data_df['num_dict_token'] < max_dict_token_num) \
    & (data_df['num_ft_text_dict_token'] < max_ft_text_token_num)
print('ft_index: ', ft_index.sum())

pt_index = (data_df['num_dict_token'] < max_dict_token_num) \
    & (data_df['num_pt_token'] < max_pt_token_num)
print('pt_index: ', pt_index.sum()) 

ft_index = topk_index(ft_index, max_ft_data_size)
intersection = ft_index & pt_index
print('intersection: ', intersection.sum())
pt_index = pt_index & (~intersection)
pt_index = topk_index(pt_index, max_pt_data_size)

pt_df = data_df[pt_index]
ft_df = data_df[ft_index]

pt_normal_df = pt_df.progress_apply(lambda r: pd.Series(dict(input=r['pt_input'],data=r['label_data'],output=r['pt_output'])), axis=1)
ft_no_dict_df = ft_df.progress_apply(lambda r: pd.Series(dict(input=r['ft_input'], output=r['ft_output'])), axis=1)
ft_text_dict_df = ft_df.progress_apply(lambda r: pd.Series(dict(input=r['ft_text_dict_input'], output=r['ft_output'])), axis=1)
ft_normal_df = ft_df.progress_apply(lambda r: pd.Series(dict(input=table_token+r['ft_input'], data=r['label_data'], output=r['ft_output'])), axis=1)
ft_raw_data_df = ft_df.progress_apply(lambda r: pd.Series(dict(input=table_token+r['ft_input'], data=r['raw_data'], output=r['ft_output'])), axis=1)

pt_size = int(len(pt_df)*(1-pt_eval_ratio))
train_df_pretrain = pt_normal_df.iloc[:pt_size]
eval_df_pretrain = pt_normal_df.iloc[pt_size:]

ft_size = int(len(ft_df)*(1-ft_eval_ratio))
train_df = ft_normal_df.iloc[:ft_size]
eval_df = ft_normal_df.iloc[ft_size:]
train_df_text_dicts = ft_text_dict_df.iloc[:ft_size]
eval_df_text_dicts = ft_text_dict_df.iloc[ft_size:]
train_df_no_dicts = ft_no_dict_df.iloc[:ft_size]
eval_df_no_dicts = ft_no_dict_df.iloc[ft_size:]
train_df_raw = ft_raw_data_df.iloc[:ft_size]
eval_df_raw = ft_raw_data_df.iloc[ft_size:]

data_df['num_dict_token'].hist(backend='plotly', title='num_dict_token').show()
pt_df['num_pt_token'].hist(backend='plotly', title='num_pt_token').show()
ft_df['num_ft_token'].hist(backend='plotly', title='num_ft_token').show()
ft_df['num_ft_text_dict_token'].hist(backend='plotly', title='num_ft_text_dict_token').show()
print(f"original diagnosis terms num: {len(term2bterm.keys())}")
print(f"normalized diagnosis terms num: {len(set(list(term2bterm.values())))}")
print(term2bterm)
bterm2terms = {}
for t, bt in term2bterm.items():
    if bt not in bterm2terms:
        bterm2terms[bt] = []
    bterm2terms[bt].append(t)

if not os.path.exists(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/")):
    os.mkdir(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/"))
json.dump(train_df.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_train.json"), 'w'), ensure_ascii=False)
json.dump(eval_df.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_eval.json"), 'w'), ensure_ascii=False, indent=4)
json.dump(train_df_no_dicts.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_train_no_dicts.json"), 'w'), ensure_ascii=False)
json.dump(eval_df_no_dicts.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_eval_no_dicts.json"), 'w'), ensure_ascii=False, indent=4)
json.dump(train_df_text_dicts.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_train_text_dicts.json"), 'w'), ensure_ascii=False)
json.dump(eval_df_text_dicts.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_eval_text_dicts.json"), 'w'), ensure_ascii=False, indent=4)
json.dump(train_df_raw.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_train_raw_data.json"), 'w'), ensure_ascii=False)
json.dump(eval_df_raw.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_eval_raw_data.json"), 'w'), ensure_ascii=False, indent=4)
json.dump(train_df_pretrain.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_train_pretrain.json"), 'w'), ensure_ascii=False)
json.dump(eval_df_pretrain.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_eval_pretrain.json"), 'w'), ensure_ascii=False, indent=4)