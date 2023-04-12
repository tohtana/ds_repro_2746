deepspeed --num_gpus=2 ./run_seq2seq_deepspeed_peft.py \
    --model_id google/flan-t5-base \
    --dataset_path data \
    --epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --generation_max_length 129 \
    --lr 1e-4 \
    --deepspeed configs/ds_flan_t5_z3_config_bf16.json \
    2>&1 | tee run_t5_debug.log
