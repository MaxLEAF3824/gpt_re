from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from huggingface_hub import snapshot_download as huggingface_snapshot_download
from modelscope import snapshot_download as modelscope_snapshot_download
import os
from model import *
model_name = "internlm/internlm-7b"

# with LoadWoInit():
#     model = AutoModelForCausalLM.from_pretrained(os.path.join(os.environ['my_models_dir'],'internlm-7b'), trust_remote_code=True).cuda()
# tok = AutoTokenizer.from_pretrained(os.path.join(os.environ['my_models_dir'],'internlm-7b'), trust_remote_code=True)

# res = model.generate(input_ids=tok("hello?",return_tensors='pt')['input_ids'].cuda())
# print('res: ', res)


mt = LLM.from_pretrained(model_path=os.path.join(os.environ['my_models_dir'],'internlm-7b'))
mt.model.cuda()
res = mt.generate("hello?")
print('res: ', res)