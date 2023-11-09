cd `dirname $0`
torchrun --nproc_per_node=1 --master_port=12570 ../hf_trainer.py \
    --mt_path $my_models_dir/llama-7b  \
    --train_data_path $my_datasets_dir/ninth/checkout_data_train.json \
    --eval_data_path $my_datasets_dir/ninth/checkout_data_eval.json \
    --output_dir ../output/dict-llama-7b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --bf16 True \
    # --gradient_checkpointing True \
