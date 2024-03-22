
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=6 \
torchrun --nproc_per_node=1 --master_port=26705 \
train_pattern_predictor.py \
   --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --output_dir ./ckpts/ \
    --bf16 True \
    --tf32 True \
    --evaluation_strategy "epoch" \
    --logging_steps 20 \
    --per_device_eval_batch_size 64 \
    --eval_only True \
    --ckpt_path /home/nus-hx/code/vllm/examples/ckpts/finetuneAll_AlltrainMax512StartRandom_EvalMax512_2Layer_bceIgnore_baselr2e-5wd1e-4_head1e-3wd5e-3/checkpoint-5625/model.safetensors \
    --do_train False \
    --do_eval True \
    --eval_max_seq_size 512 \
    --data_type 1 \
    $@