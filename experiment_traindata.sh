DATASET=PopQA # [PopQA, TriviaQA, NaturalQuestions, 2WikiMultiHopQA, ASQA]
MODEL=InstructRAG-ICL # [InstructRAG-FT, InstructRAG-ICL]
DATAPATH=../../../dataspace/P76124574/InstructRAG/
CMP1_NAME=train_inter_exter_v4
CMP2_NAME=_ # [_instrag, _inter_exter, _vanilla]

CUDA_VISIBLE_DEVICES=0 python src/experiment.py \
  --dataset_name $DATASET \
  --rag_model $MODEL \
  --datapath $DATAPATH \
  --cmp1_name $CMP1_NAME \
  --cmp2_name $CMP2_NAME \
  --experiment_date 2025_04_19_02 \
  --save True \
  --task_type check_train \
  --check_version r7 \
#   --do_inter_exter True \
  # --do_internal_generation \
  # --do_rationale_generation \
  # --max_instances 100 \
  # --do_vanilla True \

  # --load_local_model # Uncomment this line if you want to load a local model
  # output_dir ${DATAPATH}/eval_results/${MODEL}/${DATASET}