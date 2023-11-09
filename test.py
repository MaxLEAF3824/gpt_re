from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download as huggingface_snapshot_download
from modelscope import snapshot_download as modelscope_snapshot_download
import os

model_name = "internlm/internlm-7b"
# model_name = "baichuan-inc/Baichuan2-7B-Base"

local_dir = os.path.join(os.environ['my_models_dir'], model_name.split("/")[-1])
huggingface_snapshot_download(model_name, cache_dir=local_dir, local_dir=local_dir, local_dir_use_symlinks=False, ignore_patterns=["*.h5","*safetensors","*msgpack"],etag_timeout=60)
# modelscope_snapshot_download(mdeol_name, ignore_file_pattern=["*.h5","*safetensors","*msgpack"])