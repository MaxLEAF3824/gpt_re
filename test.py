from model.dict_llm import *
import os
import json

data = json.load(open(os.path.join(os.environ['my_datasets_dir'],'ninth/v3/checkout_data_eval.json')))
d = data[0]
dllm = DictLLM(mt_path=os.path.join(os.environ['my_models_dir'],'gpt2'),
               encoder_hidden_size=768,
               num_table_token=20,
               num_encoder_head=8,
               num_encoder_layers=12,
               max_length=2048,
               special_tokens_path=os.path.join(os.environ['my_datasets_dir'],'ninth/checkout_data_special_tokens.json'),
               mask_strategy="none",
               encoder_type="bert")

dllm(input_text=d['input'], dicts=d['data'])