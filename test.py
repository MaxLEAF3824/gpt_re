import json
import os
import pandas as pd
from transformers import AutoTokenizer

llm_tok = AutoTokenizer.from_pretrained(os.path.join(os.environ['my_models_dir'],'internlm-7b'), trust_remote_code=True)
data_path = os.path.join(os.environ['my_datasets_dir'], "ninth/checkout_data_with_checks.json")
eval_ratio = 0.01
max_length = 768
max_length2 = 2048

df = pd.read_json(data_path)
print('full df size: ', df.shape[0])

data_df = df.apply(lambda r: r if (r['门诊诊断'] and r['出院诊断'] and r['门诊诊断'] not in r['出院诊断']) else None,axis=1).dropna()
data_df = data_df.apply(lambda r: pd.Series(dict(
    input="性别:"+r['性别']+"\n年龄:"+r['年龄']+"\n入院时主要症状及体征:"+r['入院时主要症状及体征']+"\n特殊检查及重要会诊:"+r['特殊检查及重要会诊']+"\n出院诊断:",
    data=r['完整化验结果'],
    output=r['出院诊断'])),axis=1)
# data_df['input'] = data_df.apply(lambda r: r['input'].replace(r['output','未知']), axis=1)
print(f"filterd by 门诊and出院: {data_df.shape[0]} left")

data_df['num_tokens'] = data_df.apply(lambda r : len(llm_tok(r['input']+r['output'])['input_ids']),axis=1)
data_df = data_df[data_df['num_tokens'] < max_length]
print(f"filterd by length: {data_df.shape[0]} left")

data_df = data_df.sample(frac=1)
data_df_text_dicts = data_df.apply(lambda r: pd.Series(dict(input=r['input']+'完整化验结果: '+', '.join([f"{k}:{v.replace('[','').replace(']','')}" for dict in r['data'] for (k,v) in dict['values'].items()]),output=r['output'])), axis=1)
data_df_text_dicts['num_tokens'] = data_df_text_dicts.apply(lambda r : len(llm_tok(r['input']+r['output'])['input_ids']),axis=1)

data_df = data_df[data_df_text_dicts['num_tokens'] < max_length2]
data_df_text_dicts = data_df_text_dicts[data_df_text_dicts['num_tokens'] < max_length2]
data_df_no_dicts = data_df.drop(columns=['data'])
print(f'filterd by length 2: {data_df.shape[0]} left')

train_df = data_df.iloc[:int(len(data_df)*(1-eval_ratio))]
eval_df = data_df.iloc[int(len(data_df)*(1-eval_ratio)):]
train_df_text_dicts = data_df_text_dicts.iloc[:int(len(data_df_text_dicts)*(1-eval_ratio))]
eval_df_text_dicts = data_df_text_dicts.iloc[int(len(data_df_text_dicts)*(1-eval_ratio)):]
train_df_no_dicts = data_df_no_dicts.iloc[:int(len(data_df_no_dicts)*(1-eval_ratio))]
eval_df_no_dicts = data_df_no_dicts.iloc[int(len(data_df_no_dicts)*(1-eval_ratio)):]

json.dump(train_df.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], "ninth/checkout_data_train.json"),'w'), ensure_ascii=False)
json.dump(eval_df.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], "ninth/checkout_data_eval.json"),'w'), ensure_ascii=False)
json.dump(train_df_no_dicts.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], "ninth/checkout_data_train_no_dicts.json"),'w'), ensure_ascii=False)
json.dump(eval_df_no_dicts.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], "ninth/checkout_data_eval_no_dicts.json"),'w'), ensure_ascii=False)
json.dump(train_df_text_dicts.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], "ninth/checkout_data_train_text_dicts.json"),'w'), ensure_ascii=False)
json.dump(eval_df_text_dicts.to_dict(orient="records"), open(os.path.join(os.environ['my_datasets_dir'], "ninth/checkout_data_eval_text_dicts.json"),'w'), ensure_ascii=False)

print(data_df['num_tokens'].describe())
print(data_df_text_dicts['num_tokens'].describe())
data_df_text_dicts['num_tokens'].hist(bins=1000,log=True)