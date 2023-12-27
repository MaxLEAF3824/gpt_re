# 生成v3.2/checkout_data_train训练数据集
# 数据来源：各大内科，使用bert_score筛选与化验相关度高的数据，进一步清洗了出院诊断
# 进一步增加了stage1 Encoder预训练数据，任务：还原化验单的异常表项
import json
import os
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import re
import random


llm_tok = AutoTokenizer.from_pretrained(os.path.join(os.environ['my_models_dir'],'internlm-7b'), trust_remote_code=True)
encoder_tok = AutoTokenizer.from_pretrained(os.path.join(os.environ['my_models_dir'],'bert-base-chinese'), trust_remote_code=True)
data_path = os.path.join(os.environ['my_datasets_dir'], "ninth/checkout_data_im_with_checks.json")
ft_eval_ratio = 0.05
prtrain_eval_ratio = 0.01
bert_score_f1_threshold = 0.8
max_length1 = 768 # max length for input text
max_length2 = 1600 # max length for input text + text dicts
max_ft_data_size = 13000 # FT 训练数据集规模
seed = 42
version = "3.2"
table_token='[TABLE]'
max_length3 = 128
max_pretrain_data_size = 20000 # Pretrain 训练数据集规模
max_num_abnormal_sample = 10 # 每个化验单最多sample的异常检验项目数
pretrain_templates = [
    f"{table_token} 请输出{max_num_abnormal_sample}个化验单中包含的异常检验项目。输出:",
    f"{table_token} 请列出{max_num_abnormal_sample}个化验单中的所有异常检验结果。输出:",
    f"{table_token} 告诉我化验单里{max_num_abnormal_sample}个异常的检验项目。输出:",
    f"{table_token} 我需要知道化验单上{max_num_abnormal_sample}个不正常的检验指标。输出:",
    f"{table_token} 请展示化验单上{max_num_abnormal_sample}个异常的检测项。",
    f"{table_token} 查看化验单，并标出其中{max_num_abnormal_sample}个不正常的检验项目。输出:",
    f"{table_token} 需要了解化验单中哪些项目的检测结果异常，列出{max_num_abnormal_sample}个。输出:",
    f"{table_token} 请指出化验单中{max_num_abnormal_sample}个存在异常的检验指标。输出:",
    f"{table_token} 化验单里有哪些项目检测出现异常，请列出来{max_num_abnormal_sample}个。输出:",
    f"{table_token} 请识别并报告{max_num_abnormal_sample}个化验单上的异常检测项目。输出:",
    f"{table_token} 检查化验单，并确定哪些检验项目是异常的，列出{max_num_abnormal_sample}个。输出:"
]

random.seed(seed)
tqdm.pandas()

df = pd.read_json(data_path)
df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
print('full df size: ', df.shape[0])

data_df = df.progress_apply(
    lambda r: pd.Series(
        dict(
            input="性别:"+r['性别']+"\n年龄:"+r['年龄']+"\n入院时主要症状及体征:"+r['入院时主要症状及体征'].replace("\n","")+"\n特殊检查及重要会诊:"+r['特殊检查及重要会诊'].replace("\n","")+"\n出院诊断:",
            data=r['完整化验结果'], 
            output=r['出院诊断'],
        )
    ),
    axis=1
)

def get_abnormal_items(dict):
    abnormal_items = []
    for d in dict:
        for (k,v) in d['data'].items():
            if v in ["空", "阳性", "阳性(+)", "弱阳性", "极弱阳性"]:
                abnormal_items.append(k)
    if abnormal_items:
        num_sample = min(len(abnormal_items), max_num_abnormal_sample)
        abnormal_items = random.sample(abnormal_items, num_sample)
        return ", ".join(abnormal_items)
    else:
        return "无"

data_df_pretrain = data_df.progress_apply(
    lambda r: pd.Series(
        dict(
            input=random.choice(pretrain_templates),
            data=[[dict(header=d['header'],data=d['data']) for d in r['data']]],
            output=get_abnormal_items(r['data'])
        )
    ),
    axis=1
)

data_df_pretrain['num_tokens'] = data_df_pretrain.progress_apply(lambda r : len(llm_tok(r['input']+r['output'])['input_ids']), axis=1)
data_df_pretrain = data_df_pretrain[data_df_pretrain['num_tokens'] < max_length3]
print(f"filterd by max_length3: {data_df_pretrain.shape[0]} left")

data_df = data_df[(df['门诊诊断'].apply(len)>0)&(df['出院诊断'].apply(len)>0)&(df['门诊出院bert_score_f1']<bert_score_f1_threshold)]
print(f"filterd by 门诊 and 出院 and bert_score > {bert_score_f1_threshold}: {data_df.shape[0]} left")

# wash
def wash(r):
    new_output = r['output'].replace("「","").replace("」","").replace("\n","").replace(" ","").replace("(","（").replace(")","）")
    new_output = re.sub("（[^）]*）","",new_output)
    new_input = r['input'].replace("「","").replace("」","")
    for o in new_output.split("，"):
        new_input = new_input.replace(o,"")
    return pd.Series(dict(input=new_input, data=r['data'], output=new_output))

data_df = data_df.progress_apply(wash, axis=1)

data_df['num_tokens'] = data_df.progress_apply(lambda r : len(llm_tok(r['input']+r['output'])['input_ids']),axis=1)
data_df = data_df[data_df['num_tokens'] < max_length1]
print(f"filterd by max_length1: {data_df.shape[0]} left")

data_df_text_dicts = data_df.progress_apply(lambda r: pd.Series(
    dict(
        input='完整化验结果:'+', '.join([f"{k}:{v}" for dict in r['data'] for (k,v) in dict['data'].items()]) + "\n" + r['input'],
        output=r['output']
    )
), axis=1)
data_df_text_dicts['num_tokens'] = data_df_text_dicts.progress_apply(lambda r : len(llm_tok(r['input']+r['output'])['input_ids']),axis=1)
data_df = data_df[data_df_text_dicts['num_tokens'] < max_length2]
data_df_text_dicts = data_df_text_dicts[data_df_text_dicts['num_tokens'] < max_length2]
print(f'filterd by max_length2: {data_df.shape[0]} left')

data_df_no_dicts = data_df.drop(columns=['data'])

data_df_raw = data_df.progress_apply(lambda r: pd.Series(
    dict(
        input=table_token+r['input'],
        data=[[dict(header=d['header'],data=d['raw_data']) for d in r['data']]], 
        output=r['output'],
    )
) ,axis=1)

data_df_label = data_df.progress_apply(lambda r: pd.Series(
    dict(
        input=table_token+r['input'],
        data=[[dict(header=d['header'],data=d['data']) for d in r['data']]], 
        output=r['output'],
    )
) ,axis=1)

data_df_pretrain = data_df_pretrain.iloc[:max_pretrain_data_size]
pretrain_size = int(len(data_df_pretrain)*(1-prtrain_eval_ratio))
train_df_pretrain = data_df_pretrain.iloc[:pretrain_size]
eval_df_pretrain = data_df_pretrain.iloc[pretrain_size:]

data_df = data_df.iloc[:max_ft_data_size]
ft_size = int(len(data_df)*(1-ft_eval_ratio))
train_df = data_df_label.iloc[:ft_size]
eval_df = data_df_label.iloc[ft_size:]
train_df_text_dicts = data_df_text_dicts.iloc[:ft_size]
eval_df_text_dicts = data_df_text_dicts.iloc[ft_size:]
train_df_no_dicts = data_df_no_dicts.iloc[:ft_size]
eval_df_no_dicts = data_df_no_dicts.iloc[ft_size:]
train_df_raw = data_df_raw.iloc[:ft_size]
eval_df_raw = data_df_raw.iloc[ft_size:]

print(data_df['num_tokens'].describe())
data_df['num_tokens'].hist(backend='plotly', title='text input token num').show()
print(data_df_text_dicts['num_tokens'].describe())
data_df_text_dicts['num_tokens'].hist(backend='plotly', title='text dict token num').show()
data_df_pretrain['num_tokens'].hist(backend='plotly', title='pretrain output token num').show()

json.dump(train_df.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_train.json"),'w'), ensure_ascii=False)
json.dump(eval_df.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_eval.json"),'w'), ensure_ascii=False, indent=4)
json.dump(train_df_no_dicts.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_train_no_dicts.json"),'w'), ensure_ascii=False)
json.dump(eval_df_no_dicts.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_eval_no_dicts.json"),'w'), ensure_ascii=False, indent=4)
json.dump(train_df_text_dicts.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_train_text_dicts.json"),'w'), ensure_ascii=False)
json.dump(eval_df_text_dicts.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_eval_text_dicts.json"),'w'), ensure_ascii=False, indent=4)
json.dump(train_df_raw.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_train_raw_data.json"),'w'), ensure_ascii=False)
json.dump(eval_df_raw.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_eval_raw_data.json"),'w'), ensure_ascii=False, indent=4)
json.dump(train_df_pretrain.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_train_pretrain.json"),'w'), ensure_ascii=False)
json.dump(eval_df_pretrain.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], f"ninth/v{version}/checkout_data_eval_pretrain.json"),'w'), ensure_ascii=False, indent=4)