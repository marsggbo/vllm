CUDA_VISIBLE_DEVICES=1 python train_pattern_predictor.py \
   --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --output_dir ./ckpts/mixtral_moe_pattern_predictor \
    --bf16 True \
    --tf32 True \
    --evaluation_strategy "no" \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    $@