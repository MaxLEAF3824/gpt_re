#!/bin/bash

model_size=7B
python /nvme/guoyiqiu/miniconda3/envs/hug42/lib/python3.8/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /nvme/guoyiqiu/llama \
    --model_size $model_size \
    --output_dir /nvme/guoyiqiu/llama/llama_$model_size\_hf
