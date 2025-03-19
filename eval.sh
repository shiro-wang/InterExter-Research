DATASET=PopQA # [PopQA, TriviaQA, NaturalQuestions, 2WikiMultiHopQA, ASQA]
MODEL=InstructRAG-ICL # [InstructRAG-FT, InstructRAG-ICL]
DATAPATH=../../../dataspace/P76124574/InstructRAG/

CUDA_VISIBLE_DEVICES=3 python src/inference.py \
  --dataset_name $DATASET \
  --rag_model $MODEL \
  --n_docs 5 \
  --output_dir ${DATAPATH}/eval_results/${MODEL}/${DATASET} \
  --datapath $DATAPATH \
  --input_file test \
  --output_file demos_inter_exter \
  # --do_internal_generation \
  # --do_inter_exter True \
  # --version v4 \
  # --do_rationale_generation \
  # --do_vanilla True \
  # --max_instances 100 \
  # --do_internal_generation \
  # --lost_in_mid True \
  # --load_local_model # Uncomment this line if you want to load a local model
  # output_dir ${DATAPATH}/eval_results/${MODEL}/${DATASET}