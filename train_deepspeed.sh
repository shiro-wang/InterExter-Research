DATASET=PopQA
PER_DEVICE_BATCH_SIZE=1
NUM_DEVICE=2
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_DEVICE/$PER_DEVICE_BATCH_SIZE))
DS_SKIP_CUDA_CHECK=1
DATAPATH=../../../dataspace/P76124574/InstructRAG/
DEEPSPEED_CONFIG_PATH=configs/deepspeed_config.json

CUDA_VISIBLE_DEVICES="2,3" torchrun --nproc_per_node=$NUM_DEVICE src/finetune.py \
  --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset_name $DATASET \
  --train_file_name train_inter_exter_v4_r7_icl \
  --output_dir ${DATAPATH}/saved_checkpoints/InstructRAG-FT/${DATASET}_z2_b256_e2_4096_lora_r7_icl_v2 \
  --num_train_epochs 2 \
  --n_docs 5 \
  --logging_steps 1 \
  --seed 42 \
  --bf16 True \
  --learning_rate 2.5e-5 \
  --lr_scheduler_type "cosine" \
  --warmup_ratio 0.03 \
  --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --model_max_length 4096 \
  --internal True \
  --deepspeed $DEEPSPEED_CONFIG_PATH \
  --gradient_checkpointing \