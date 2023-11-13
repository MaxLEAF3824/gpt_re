from model import DictLLM
import os.path as osp
import os
import json

mt_path = osp.join(os.environ['my_models_dir'],'internlm-7b')
data_path = osp.join(os.environ['my_datasets_dir'],'ninth/checkout_data_eval_no_dicts.json')
data = json.load(open(data_path))
dllm = DictLLM(mt_path, 64, 1, 8, 1, 2048).cuda()

for d in data:
    print('d[input]: ', d['input'])
    output_text = dllm.generate(input_text=[d['input']], max_new_tokens=20, cut_input=True)
    print(output_text)