# WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,3,6,7 torchrun train_pattern_predictor.py \

# WANDB_MODE=online CUDA_VISIBLE_DEVICES=1 \
# python -m ipdb \
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=6 \
torchrun --nproc_per_node=1 --master_port=26705 \
train_pattern_predictor.py \
   --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --output_dir ./ckpts/ \
    --bf16 True \
    --tf32 True \
    --evaluation_strategy "epoch" \
    --lazy_preprocess True \
    --save_strategy "epoch" \
    --save_steps 500 \
    --save_total_limit 1 \
    --logging_steps 20 \
    --num_train_epochs 40 \
    --load_best_model_at_end True \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --weight_decay 5e-3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    $@
   #  --sharded_ddp "simple" \
