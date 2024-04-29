CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path ../glm2 \
    --adapter_name_or_path ./save_ckpts/train_720_from_expert_revision_r3/checkpoint-157 \
    --dataset train_52k_alpaca_original\
    --template chatglm2 \
    --finetuning_type lora \
    --output_dir infer_result/train_52k_alpaca_original \
    --per_device_eval_batch_size 12 \
    --max_samples 2800 \
    --predict_with_generate \
    --fp16\
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_target "query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
