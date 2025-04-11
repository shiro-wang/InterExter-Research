import os
import sys
import argparse
import data_utils
import common_utils

from metrics import get_metrics
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
from vllm.lora.request import LoRARequest

def generate_rationale(args):
    data_path = args.datapath + f'eval_results/{args.rag_model}/{args.dataset_name}/with_internal/{args.input_file}.json' # f'dataset/{args.dataset_name}/train.json'
    print(f"Loading training set from: {data_path}")
    train_data = common_utils.jload(data_path)[:args.max_instances]

    llm = LLM(model=args.model_name_or_path, download_dir=args.cache_dir, max_model_len=args.max_tokens)

    tokenizer = llm.get_tokenizer()
    prompt_dict = common_utils.jload(args.prompt_dict_path)

    prompts = data_utils.format_prompt_with_data_list(
        args=args,
        data_list=train_data,
        dataset_name=args.dataset_name,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        n_docs=args.n_docs,
        do_rationale_generation=True,
        do_inter_exter=args.do_inter_exter,
        lost_in_mid=args.lost_in_mid,
    )

    sampling_params = SamplingParams(temperature=args.temperature, 
                                    max_tokens=args.max_tokens, 
                                    seed=args.seed,
                                    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    
    outputs = llm.generate(prompts, sampling_params)

    if args.do_inter_exter:
        output_file = os.path.join(args.output_dir, f"with_rationale/{args.output_file}_{args.version}.json")
    else:
        output_file = os.path.join(args.output_dir, "with_rationale/train.json")

    save_outputs(outputs, train_data, output_file, args.n_docs, with_internal=args.do_inter_exter)

def generate_internal_knowledge(args):
    data_path = args.datapath + f'dataset/{args.dataset_name}/{args.input_file}.json'
    print(f"Loading dataset from: {data_path}")
    train_data = common_utils.jload(data_path)[:args.max_instances]

    llm = LLM(model=args.model_name_or_path, download_dir=args.cache_dir, max_model_len=args.max_tokens)

    tokenizer = llm.get_tokenizer()
    prompt_dict = common_utils.jload(args.prompt_dict_path)

    prompts = data_utils.format_prompt_with_data_list(
        args=args,
        data_list=train_data,
        dataset_name=args.dataset_name,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        n_docs=args.n_docs,
        do_rationale_generation=False,
        do_internal_generation=True,
        do_inter_exter=True,
        lost_in_mid=args.lost_in_mid,
    )

    sampling_params = SamplingParams(temperature=args.temperature, 
                                    max_tokens=args.max_tokens, 
                                    seed=args.seed,
                                    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    
    outputs = llm.generate(prompts, sampling_params)

    output_file = os.path.join(args.output_dir, f"with_internal/{args.output_file}_{args.version}.json")

    save_outputs(outputs, train_data, output_file, args.n_docs, save_internal=True,)
   
def eval_model(args):
    if args.do_inter_exter:
        data_path = args.datapath + f'eval_results/InstructRAG-ICL/{args.dataset_name}/with_internal/{args.input_file}.json' # f'dataset/{args.dataset_name}/test.json' # f'eval_results/{args.rag_model}/{args.dataset_name}/with_internal/test_inter_exter.json'
    else:
        data_path = args.datapath + f'dataset/{args.dataset_name}/test.json'
    print(f"Loading eval set from: {data_path}")
    test_data = common_utils.jload(data_path)[:args.max_instances]

    print(f'Loading model {args.rag_model}...')
    demos = []
    if args.rag_model == 'InstructRAG-FT':
        if args.load_local_model:
            if args.lora:
                lora_path = args.datapath + f'saved_checkpoints/InstructRAG-FT/{args.dataset_name}_{args.ft_model_id}'
                llm = LLM(model=args.model_name_or_path, download_dir=args.cache_dir, max_model_len=args.max_tokens, enable_lora=True)
            else:
                llm = LLM(model=args.datapath + f'saved_checkpoints/InstructRAG-FT/{args.dataset_name}_{args.ft_model_id}',  max_model_len=args.max_tokens)
        else:
            llm = LLM(model=f'meng-lab/{args.dataset_name}-InstructRAG-FT', download_dir=args.cache_dir, max_model_len=args.max_tokens)
    elif args.rag_model == 'InstructRAG-ICL':
        if not args.do_vanilla:
            if args.do_inter_exter:
                demos = common_utils.jload(args.datapath + f'eval_results/{args.rag_model}/{args.dataset_name}/with_rationale/demos_inter_exter_{args.demo_version}.json')
            else:
                demos = common_utils.jload(args.datapath + f'dataset/{args.dataset_name}/demos.json') # f'dataset/{args.dataset_name}/demos.json' # f'eval_results/{args.rag_model}/{args.dataset_name}/with_rationale/demos_inter_exter.json')
        llm = LLM(model=args.model_name_or_path, download_dir=args.cache_dir, max_model_len=args.max_tokens)

    tokenizer = llm.get_tokenizer()
    prompt_dict = common_utils.jload(args.prompt_dict_path)
 
    prompts = data_utils.format_prompt_with_data_list(
        args=args,
        data_list=test_data,
        dataset_name=args.dataset_name,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        n_docs=args.n_docs,
        demos=demos,
        do_inter_exter=args.do_inter_exter,
        do_vanilla = args.do_vanilla,
        lost_in_mid=args.lost_in_mid,
    )
    
    sampling_params = SamplingParams(temperature=args.temperature, 
                                    max_tokens=args.max_tokens, 
                                    seed=args.seed,
                                    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    if args.lora:
        outputs = llm.generate(prompts, sampling_params, lora_request=LoRARequest("popqa_adapter", 1, lora_path))
    outputs = llm.generate(prompts, sampling_params)
    
    if args.do_inter_exter:
        output_file = os.path.join(args.output_dir, f"{args.output_file}_{args.version}.json")
    elif args.do_vanilla:
        output_file = os.path.join(args.output_dir, "result_vanilla.json")
    else:
        output_file = os.path.join(args.output_dir, "result_instrag.json")

    eval_results = save_outputs(outputs, test_data, output_file, args.n_docs)
    get_metrics(args, eval_results, args.output_dir, is_asqa=args.dataset_name == 'ASQA', with_internal=args.do_inter_exter, vanilla=args.do_vanilla)

def save_outputs(outputs, test_data, output_file, n_docs, save_internal=False, with_internal=False):
    # Save the outputs as a JSON file.
    output_data = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        sample = test_data[i]
        if save_internal:
            output_data.append({
                "question": sample["question"],
                "answers": sample["answers"],
                "qa_pairs": sample["qa_pairs"] if "qa_pairs" in sample else None,
                "prompt": prompt,
                "ctxs": sample["ctxs"][:n_docs][::-1] if (sample["ctxs"][0]['score'] > sample["ctxs"][1]['score']) else sample["ctxs"][:n_docs],
                "internal_knowledge": generated_text,
                })
        else:
            output_data.append({
                "question": sample["question"],
                "answers": sample["answers"],
                "qa_pairs": sample["qa_pairs"] if "qa_pairs" in sample else None,
                "rationale": generated_text,
                "prompt": prompt,
                "ctxs": sample["ctxs"][:n_docs][::-1] if (sample["ctxs"][0]['score'] > sample["ctxs"][1]['score']) else sample["ctxs"][:n_docs],
                })
            if with_internal:
                output_data[-1]["internal_knowledge"] = sample["internal_knowledge"]
        
    common_utils.jdump(output_data, output_file)
    print(f"Outputs saved to {output_file}")

    return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--datapath', type=str, help='Path to the dataset', default='')
    parser.add_argument('--input_file', type=str, help='Path to the input file')
    parser.add_argument('--output_file', type=str, help='Path to the output file')
    parser.add_argument('--rag_model', type=str, choices=['InstructRAG-FT', 'InstructRAG-ICL'], default='InstructRAG-FT', help='InstructRAG model: InstructRAG-FT or InstructRAG-ICL')
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='name of the model in Hugging Face model hub or path to the model')
    parser.add_argument('--load_local_model', action='store_true', help='Load local model')
    parser.add_argument('--lora', action='store_true', help='Load LoRA adapter, base model is the same as model_name_or_path, lora_path is same as ft_model_id')
    parser.add_argument('--ft_model_id', type=str, default='z2_b256_e2_4096', help='Fine-tuned model ID')
    parser.add_argument('--n_docs', type=int, default=5, help='Number of retrieved documents')
    parser.add_argument('--lost_in_mid', type=bool, default=False, help='Lost in the middle problem in RAG => reordering the documents')
    parser.add_argument('--output_dir', type=str, help='Path to the output file')
    parser.add_argument('--cache_dir', type=str, default='../../../dataspace/P76124574/InstructRAG/models/', help='Directory to cached models')
    parser.add_argument('--prompt_dict_path', type=str, default="src/rag.json")
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for sampling')
    parser.add_argument('--max_tokens', type=int, default=4096, help='Maximum number of tokens')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_instances', type=int, default=sys.maxsize)
    parser.add_argument('--do_internal_generation', action='store_true', help='Generate internal knowledge')
    parser.add_argument('--do_rationale_generation', action='store_true', help='Generate rationales')
    parser.add_argument('--do_rationale_generation_predefined', action='store_true', help='Generate rationales on predefined data')
    parser.add_argument('--do_inter_exter', type=bool, default=False, help='Generate internal and external knowledge')
    parser.add_argument('--do_vanilla', type=bool, default=False, help='Generate vanilla outputs')
    parser.add_argument('--version', type=str, default='', help='Version of the dataset')
    parser.add_argument('--demo_version', type=str, default='', help='Version of the demos')

    args = parser.parse_args()
    
    if args.do_rationale_generation:
        print("Generating rationales...")
        generate_rationale(args)
    elif args.do_internal_generation:
        print("Generating internal knowledge...")
        generate_internal_knowledge(args)
    else:
        print("Evaluating model...")
        eval_model(args)
