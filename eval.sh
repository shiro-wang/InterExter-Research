DATASET=TriviaQA # [PopQA, TriviaQA, NaturalQuestions, 2WikiMultiHopQA, ASQA]
MODEL=InstructRAG-ICL # [InstructRAG-FT, InstructRAG-ICL]
DATAPATH=../../../dataspace/P76124574/InstructRAG/

CUDA_VISIBLE_DEVICES=1 python src/inference.py \
  --dataset_name $DATASET \
  --rag_model $MODEL \
  --n_docs 5 \
  --output_dir ${DATAPATH}/eval_results/${MODEL}/${DATASET} \
  --datapath $DATAPATH \
  --input_file test_inter_exter_v4 \
  --output_file result_inter_exter \
  --do_inter_exter True \
  --version v4_r3_d3 \
  --demo_version v4_r3 \
  # --do_rationale_generation \
  # --do_vanilla True \
  # --max_instances 100 \
  # --do_internal_generation \
  # --lost_in_mid True \
  # --load_local_model # Uncomment this line if you want to load a local model
  # output_dir ${DATAPATH}/eval_results/${MODEL}/${DATASET}