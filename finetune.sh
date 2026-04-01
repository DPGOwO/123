# export CUDA_VISIBLE_DEVICES=4,5,6,7
NPROC_PER_NODE=8

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=/cache/wx1427092/Qwen2.5-VL-7B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=5e-6
batch_size=4
grad_accum_steps=1

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=mydata

# Output configuration
run_name="qwen3vl"
output_dir=/cache/wx1427092/prompt_val23  #TODO
# output_dir=/cache/wx1427092/ROUTER
logging_dir="${output_dir}/log"

# Training arguments
# export WANDB_DISABLED=true
# export WANDB_MODE=disabled
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 2 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard \
    --logging_dir ${logging_dir}"
    # --lora_enable True \
    # --lora_r 128 \
    # --lora_alpha 256 \
    # --lora_dropout 0.05 \
    # --mm_projector_lr 5e-5" 

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
