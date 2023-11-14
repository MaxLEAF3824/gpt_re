cd `dirname $0`
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=12570 ../../hf_trainer.py \
    --mt_path $my_models_dir/internlm-7b  \
    --encoder_hidden_size 2 \
    --num_table_token 1 \
    --num_encoder_head 2 \
    --num_encoder_layers 1 \
    --train_data_path $my_datasets_dir/ninth/checkout_data_train_text_dicts.json \
    --eval_data_path $my_datasets_dir/ninth/checkout_data_eval_text_dicts.json \
    --output_dir $my_models_dir/bl-internlm-7b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps 1 \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --logging_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'InternLMDecoderLayer' \
    --fp16 True \