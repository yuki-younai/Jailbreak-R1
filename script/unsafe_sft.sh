export CUDA_VISIBLE_DEVICES=5,6


OUTPUT_DIR=./model_output/unsafe_sft
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
num_processes=$(expr length "$CUDA_VISIBLE_DEVICES" / 2 + 1)

target_model_path=$1

accelerate launch --main_process_port 29502 --config_file=script/accelerate_configs/zero3.yaml \
            --num_processes=$num_processes src/algorithm/unsafe_sft.py \
            --model_name_or_path $target_model_path \
            --dataset_name datasets/unsafe_data \
            --torch_dtype "bfloat16" \
            --attn_implementation flash_attention_2 \
            --use_peft False \
            --bf16 True \
            --load_in_8bit False \
            --load_in_4bit False \
            --num_train_epochs 3 \
            --max_seq_length 2048 \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 2 \
            --gradient_checkpointing True \
            --learning_rate 2e-6 \
            --optim "adamw_torch" \
            --lr_scheduler_type "constant" \
            --dataset_text_field sft \
            --warmup_ratio 0.1 \
            --logging_steps 1 \
            --save_strategy "epoch" \
            --report_to none \
            --wandb_project "jailbreak_r1" \
            --output_dir $OUTPUT_DIR 



























