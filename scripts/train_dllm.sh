#!/bin/bash
#SBATCH --job-name=Full_Train_DLLM
#SBATCH --partition=medai
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=auto
#SBATCH --output=slurm_log/%x-%j.out
#SBATCH --error=slurm_log/%x-%j.out

export WANDB_PROJECT="dicts_llm"
cd `dirname $0`
cd ..

VERSION=2
GPU_NUM=8
MASTER_PORT=12570
MODEL_NAME="internlm-7b"
SHARED_ARGS="--mt_path $my_models_dir/$MODEL_NAME  \
            --special_tokens_path $my_datasets_dir/ninth/checkout_data_special_tokens.json \
            --num_train_epochs 10 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --dataloader_num_workers 0 \
            --gradient_accumulation_steps 1 \
            --save_strategy 'steps' \
            --save_steps  0.3 \
            --save_total_limit 3 \
            --evaluation_strategy 'steps' \
            --eval_steps 500 \
            --logging_steps 1 \
            --learning_rate 1e-5 \
            --weight_decay 0. \
            --lr_scheduler_type 'cosine' \
            --fsdp 'full_shard auto_wrap' \
            --fsdp_transformer_layer_cls_to_wrap 'InternLMDecoderLayer' \
            --fp16 True \
            --seed 42"

# # CONTROL
# torchrun --nproc_per_node=$GPU_NUM --master_port=$MASTER_PORT hf_trainer.py $SHARED_ARGS \
#     --run_name control-v$VERSION \
#     --encoder_hidden_size 2 \
#     --num_table_token 1 \
#     --num_encoder_head 2 \
#     --num_encoder_layers 1 \
#     --train_data_path $my_datasets_dir/ninth/checkout_data_train_no_dicts.json \
#     --eval_data_path $my_datasets_dir/ninth/checkout_data_eval_no_dicts.json \
#     --output_dir $my_models_dir/ct-$MODEL_NAME-v$VERSION \

# # BASELINE
# torchrun --nproc_per_node=$GPU_NUM --master_port=$MASTER_PORT hf_trainer.py $SHARED_ARGS \
#     --run_name baseline-v$VERSION \
#     --encoder_hidden_size 2 \
#     --num_table_token 1 \
#     --num_encoder_head 2 \
#     --num_encoder_layers 1 \
#     --train_data_path $my_datasets_dir/ninth/checkout_data_train_text_dicts.json \
#     --eval_data_path $my_datasets_dir/ninth/checkout_data_eval_text_dicts.json \
#     --output_dir $my_models_dir/bl-$MODEL_NAME-v$VERSION \

# NAIVE
torchrun --nproc_per_node=$GPU_NUM --master_port=$MASTER_PORT hf_trainer.py $SHARED_ARGS \
    --run_name naive-v$VERSION \
    --encoder_hidden_size 768 \
    --num_table_token 10 \
    --num_encoder_head 8 \
    --num_encoder_layers 12 \
    --train_data_path $my_datasets_dir/ninth/checkout_data_train.json \
    --eval_data_path $my_datasets_dir/ninth/checkout_data_eval.json \
    --output_dir $my_models_dir/nv-$MODEL_NAME-v$VERSION \
    --no_mask True \

# # OURS
# torchrun --nproc_per_node=$GPU_NUM --master_port=$MASTER_PORT hf_trainer.py $SHARED_ARGS \
#     --run_name ours-v$VERSION \
#     --encoder_hidden_size 768 \
#     --num_table_token 10 \
#     --num_encoder_head 8 \
#     --num_encoder_layers 12 \
#     --train_data_path $my_datasets_dir/ninth/checkout_data_train.json \
#     --eval_data_path $my_datasets_dir/ninth/checkout_data_eval.json \
#     --output_dir $my_models_dir/ours-$MODEL_NAME-v$VERSION \

