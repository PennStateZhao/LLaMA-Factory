CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path ../glm2 \
    --adapter_name_or_path ./save_ckpts/epoch_10_whole_train_720_v2_lr4en4/checkpoint-330 \
    --dataset infer_520_from_train_720_v2\
    --template chatglm2 \
    --finetuning_type lora \
    --output_dir infer_result/infer_520_from_train_720_v2_lr4en4_step330 \
    --per_device_eval_batch_size 12 \
    --max_samples 2815 \
    --predict_with_generate \
    --fp16\
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_target "query_key_value,dense,dense_h_to_4h,dense_4h_to_h"

