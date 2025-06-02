export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


OUTPUT_DIR=./model_output/jailbreak_r1_train
current_time=$(date "+%Y%m%d%H%M%S")  
OUTPUT_DIR="${OUTPUT_DIR}_data${current_time}"
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
True_DIR="$(dirname "${ROOT_DIR}")"
export PYTHONPATH="${True_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

SCRIPT_NAME=$(basename "$0")
DESTINATION_PATH="$OUTPUT_DIR/$SCRIPT_NAME"
cp "$0" "$DESTINATION_PATH"

num_processes=$(expr length "$CUDA_VISIBLE_DEVICES" / 2 + 1 - 2)

warm_up_model_path=$1
unsafe_model_path=$2
classify_model_path=$3
harmbench_model_path=$4


accelerate launch  --main_process_port 29501 --config_file=script/accelerate_configs/zero3.yaml \
            --num_processes=$num_processes   src/algorithm/grpo_train_jailbreak.py \
            --model_name_or_path $warm_up_model_path \
            --dataset_name datasets/Attack_target/Train-data \
            --torch_dtype "bfloat16" \
            --attn_implementation flash_attention_2 \
            --use_peft False \
            --bf16 True \
            --load_in_8bit False \
            --load_in_4bit False \
            --do_eval False \
            --num_train_epochs 1 \
            --max_prompt_length 512 \
            --max_completion_length 1024 \
            --per_device_train_batch_size 6\
            --gradient_accumulation_steps 4 \
            --num_generations 6 \
            --num_iterations 1 \
            --gradient_checkpointing True \
            --learning_rate 1e-6 \
            --optim "adamw_torch" \
            --lr_scheduler_type constant \
            --warmup_ratio 0.01 \
            --logging_steps 1 \
            --log_completions True \
            --save_strategy "steps" \
            --save_steps 100 \
            --beta 0.01 \
            --template r1 \
            --scale_rewards True \
            --reward_funcs "warm_up_reward" \
            --reward_weights 1.0\
            --classify_model $classify_model_path \
            --target_model $unsafe_model_path \
            --judge_model $harmbench_model_path \
            --use_vllm True \
            --vllm_device auto \
            --vllm_dtype auto \
            --temperature 1.0 \
            --vllm_gpu_memory_utilization 0.5 \
            --report_to wandb \
            --wandb_project "jailbreak_r1_train" \
            --seed 42 \
            --output_dir $OUTPUT_DIR 





















































