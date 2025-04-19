DATASET=PopQA # [PopQA, TriviaQA, NaturalQuestions, 2WikiMultiHopQA, ASQA]
MODEL=InstructRAG-FT # [InstructRAG-FT, InstructRAG-ICL]
LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
DATAPATH=../../../dataspace/P76124574/InstructRAG/

CUDA_VISIBLE_DEVICES=3 python src/inference.py \
  --dataset_name $DATASET \
  --rag_model $MODEL \
  --n_docs 5 \
  --output_dir ${DATAPATH}/eval_results/${MODEL}/${DATASET} \
  --datapath $DATAPATH \
  --input_file test_inter_exter_v4 \
  --output_file result_inter_exter \
  --version v4_r7_icl \
  --demo_version r7 \
  --do_inter_exter True \
  --model_name_or_path $LLM_MODEL \
  --ft_model_id z2_b256_e2_4096_lora_r7_icl \
  --load_local_model \
  --lora 
  # --do_rationale_generation_predefined \
  # --do_rationale_generation \
  # --do_rationale_generation_icl \
  # --max_instances 100 \
  # --do_vanilla True \
  # --max_instances 100 \
  # --do_internal_generation \
  # --lost_in_mid True \
  # --load_local_model # Uncomment this line if you want to load a local model
  # output_dir ${DATAPATH}/eval_results/${MODEL}/${DATASET}

  ### lora training
  # --do_rationale_generation_predefined \
  # --do_rationale_generation \
  # --do_rationale_generation_icl \

  ### lora inference
  # --ft_model_id PopQA_z2_b256_e2_4096_lora_r7_icl \
  # --load_local_model \
  # --lora 