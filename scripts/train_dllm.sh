cd `dirname $0`
torchrun --nproc_per_node=4 --master_port=12570 ../hf_trainer.py \
    --mt_path $my_models_dir/internlm-7b  \
    --encoder_hidden_size 768 \
    --num_table_token 10 \
    --num_encoder_head 8 \
    --num_encoder_layers 12 \
    --train_data_path $my_datasets_dir/ninth/checkout_data_train.json \
    --eval_data_path $my_datasets_dir/ninth/checkout_data_eval.json \
    --output_dir ../output/dict-internlm-7b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps 1 \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'InternLMDecoderLayer' \
    --model_max_length 2048 \
    --bf16 True \
    # --gradient_checkpointing True \
