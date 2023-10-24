#!/bin/bash
#SBATCH --job-name=uc_test
#SBATCH --partition=GPU-8A100
#SBATCH --nodes=4
#SBATCH --qos=gpu_8a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=20G
#SBATCH --output=/home/cs/yangyuchen/guoyiqiu/stanford_alpaca-main/train_log_uc_9.6.txt
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=29573

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

model_name_or_path=/home/cs/yangyuchen/yushengliao/Medical_LLM/vicuna-7b/
output_dir=/home/cs/yangyuchen/guoyiqiu/model/uc_9.6/
if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi
data_path="/home/cs/yangyuchen/guoyiqiu/stanford_alpaca-main/dataset/train_uc_data.json"

torchrun --nproc_per_node=4  train.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path\
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --tf32 True
