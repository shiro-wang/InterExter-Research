import os
import sys
import argparse
import data_utils
import common_utils
from metrics import get_metrics
from vllm import LLM, SamplingParams

def confusion_matrix(cmp1, cmp2):
    both_correct = []
    both_wrong = []
    cmp1_correct_cmp2_wrong = []
    cmp2_correct_cmp1_wrong = []
    for i in range(len(cmp1)):
        if cmp1[i] == 1 and cmp2[i] == 1:
            both_correct.append(i)
        elif cmp1[i] == 0 and cmp2[i] == 0:
            both_wrong.append(i)
        elif cmp1[i] == 1 and cmp2[i] == 0:
            cmp1_correct_cmp2_wrong.append(i)
        elif cmp1[i] == 0 and cmp2[i] == 1:
            cmp2_correct_cmp1_wrong.append(i)
    print(f"Both correct: {len(both_correct)}")
    print(f"Both wrong: {len(both_wrong)}")
    print(f"{args.cmp1_name} correct, {args.cmp2_name} wrong: {len(cmp1_correct_cmp2_wrong)}")
    print(f"{args.cmp2_name} correct, {args.cmp1_name} wrong: {len(cmp2_correct_cmp1_wrong)}")
    print(f"Total: {len(cmp1)}")
    
    # test_id = 99
    # print(f"For the test id {test_id}:{args.cmp1_name}: {cmp1[test_id]}, {args.cmp2_name}: {cmp2[test_id]}")
    
    if args.save:
        common_utils.jdump({"both_correct": len(both_correct), "both_wrong": len(both_wrong), f"{args.cmp1_name}_correct_{args.cmp2_name}_wrong": len(cmp1_correct_cmp2_wrong), f"{args.cmp2_name}_correct_{args.cmp1_name}_wrong": len(cmp2_correct_cmp1_wrong)}, \
            f"{args.datapath}experiments/{args.rag_model}/{args.dataset_name}/exp_{args.experiment_date}/confusion_matrix.json")
    return both_correct, both_wrong, cmp1_correct_cmp2_wrong, cmp2_correct_cmp1_wrong
    

def check_diff(args):
    data_path1_metric = args.datapath + f'eval_results/{args.rag_model}/{args.dataset_name}/metrics{args.cmp1_name}.json' # f'dataset/{args.dataset_name}/train.json'
    data_path2_metric = args.datapath + f'eval_results/{args.rag_model}/{args.dataset_name}/metrics{args.cmp2_name}.json' # f'dataset/{args.dataset_name}/train.json'
    data_path1_result = args.datapath + f'eval_results/{args.rag_model}/{args.dataset_name}/result{args.cmp1_name}.json'
    data_path2_result = args.datapath + f'eval_results/{args.rag_model}/{args.dataset_name}/result{args.cmp2_name}.json'
    print(f"Loading dataset from: metrics{args.cmp1_name} & metrics{args.cmp2_name}")
    cmp1_metric = common_utils.jload(data_path1_metric)
    cmp2_metric = common_utils.jload(data_path2_metric)
    
    both_correct, both_wrong, cmp1_correct_cmp2_wrong, cmp2_correct_cmp1_wrong = confusion_matrix(cmp1_metric["acc_list"][:args.max_instances], cmp2_metric["acc_list"][:args.max_instances])
    
    print(f"Loading dataset from: result{args.cmp1_name} & result{args.cmp2_name}")
    cmp1_result = common_utils.jload(data_path1_result)
    cmp2_result = common_utils.jload(data_path2_result)
    # print(f"result{args.cmp1_name} correct & result{args.cmp2_name} wrong example:")
    # for i in cmp1_correct_cmp2_wrong[:2]:
    #     print(f"Example {i}:")
    #     print(f"answer: {cmp1_result[i]['answers']}")
    #     print(f"{args.cmp1_name}: {cmp1_result[i]['rationale']}")
    #     # print(f"cmp1 prompt: {cmp1_result[i]['prompt']}")
    #     print(f"{args.cmp2_name}: {cmp2_result[i]['rationale']}")
    #     # print(f"cmp2 prompt: {cmp2_result[i]['prompt']}")
    #     print("")
    # print(f"result{args.cmp2_name} correct & result{args.cmp1_name} wrong example:")
    # for i in cmp2_correct_cmp1_wrong[:1]:
    #     print(f"Example {i}:")
    #     print(f"cmp1: {cmp1_result[i]["rationale"]}")
    #     print(f"cmp2: {cmp2_result[i]["rationale"]}")
    #     print("")
    both_correct_result = [{"question_id": both_correct[i], "question": cmp1_result[both_correct[i]]["question"], "answers": cmp1_result[both_correct[i]]["answers"], \
        f"{args.cmp1_name}": cmp1_result[both_correct[i]]["rationale"], f"{args.cmp1_name}_prompt": cmp1_result[both_correct[i]]["prompt"], \
            f"{args.cmp2_name}": cmp2_result[both_correct[i]]["rationale"],  f"{args.cmp2_name}_prompt": cmp2_result[both_correct[i]]["prompt"]} for i in range(len(both_correct))]
    both_wrong_result = [{"question_id": both_wrong[i], "question": cmp1_result[both_wrong[i]]["question"], "answers": cmp1_result[both_wrong[i]]["answers"], \
        f"{args.cmp1_name}": cmp1_result[both_wrong[i]]["rationale"], f"{args.cmp1_name}_prompt": cmp1_result[both_wrong[i]]["prompt"], \
            f"{args.cmp2_name}": cmp2_result[both_wrong[i]]["rationale"],  f"{args.cmp2_name}_prompt": cmp2_result[both_wrong[i]]["prompt"]} for i in range(len(both_wrong))]
    cmp1_correct_cmp2_wrong_result = [{"question_id": cmp1_correct_cmp2_wrong[i], "question": cmp1_result[cmp1_correct_cmp2_wrong[i]]["question"], "answers": cmp1_result[cmp1_correct_cmp2_wrong[i]]["answers"], \
        f"{args.cmp1_name}": cmp1_result[cmp1_correct_cmp2_wrong[i]]["rationale"], f"{args.cmp1_name}_prompt": cmp1_result[cmp1_correct_cmp2_wrong[i]]["prompt"], \
            f"{args.cmp2_name}": cmp2_result[cmp1_correct_cmp2_wrong[i]]["rationale"],  f"{args.cmp2_name}_prompt": cmp2_result[cmp1_correct_cmp2_wrong[i]]["prompt"]} for i in range(len(cmp1_correct_cmp2_wrong))]
    cmp2_correct_cmp1_wrong_result = [{"question_id": cmp2_correct_cmp1_wrong[i], "question": cmp1_result[cmp2_correct_cmp1_wrong[i]]["question"], "answers": cmp1_result[cmp2_correct_cmp1_wrong[i]]["answers"], \
        f"{args.cmp1_name}": cmp1_result[cmp2_correct_cmp1_wrong[i]]["rationale"], f"{args.cmp1_name}_prompt": cmp1_result[cmp2_correct_cmp1_wrong[i]]["prompt"], \
            f"{args.cmp2_name}": cmp2_result[cmp2_correct_cmp1_wrong[i]]["rationale"],  f"{args.cmp2_name}_prompt": cmp2_result[cmp2_correct_cmp1_wrong[i]]["prompt"]} for i in range(len(cmp2_correct_cmp1_wrong))]
    if args.save:
        common_utils.jdump(both_correct_result, f"{args.datapath}experiments/{args.rag_model}/{args.dataset_name}/exp_{args.experiment_date}/both_correct.json")
        common_utils.jdump(both_wrong_result, f"{args.datapath}experiments/{args.rag_model}/{args.dataset_name}/exp_{args.experiment_date}/both_wrong.json")
        common_utils.jdump(cmp1_correct_cmp2_wrong_result, f"{args.datapath}experiments/{args.rag_model}/{args.dataset_name}/exp_{args.experiment_date}/{args.cmp1_name}_correct_{args.cmp2_name}_wrong.json")
        common_utils.jdump(cmp2_correct_cmp1_wrong_result, f"{args.datapath}experiments/{args.rag_model}/{args.dataset_name}/exp_{args.experiment_date}/{args.cmp2_name}_correct_{args.cmp1_name}_wrong.json")
        print(f"Saved data to the path {args.datapath}experiments/{args.rag_model}/{args.dataset_name}/exp_{args.experiment_date}")

def check_train(args):
    rationale_data_path = args.datapath + f'eval_results/{args.rag_model}/{args.dataset_name}/with_rationale/{args.cmp1_name}_{args.check_version}.json'
    gd_data_path = args.datapath + f'eval_results/{args.rag_model}/{args.dataset_name}/with_rationale/{args.cmp1_name}_gd.json'
    print(f"Loading dataset from: {args.cmp1_name}_r6.json & {args.cmp1_name}_gd.json")
    cmp1_rationale = common_utils.jload(rationale_data_path)
    cmp1_gd = common_utils.jload(gd_data_path)
    both_correct_data = []
    both_wrong_data = []
    inter_correct_data = []
    exter_correct_data = []
    for id in cmp1_gd["condition_list"]["both_correct"]:
        both_correct_data.append(cmp1_rationale[id])
    for id in cmp1_gd["condition_list"]["both_wrong"]:
        both_wrong_data.append(cmp1_rationale[id])
    for id in cmp1_gd["condition_list"]["inter_correct"]:
        inter_correct_data.append(cmp1_rationale[id])
    for id in cmp1_gd["condition_list"]["exter_correct"]:
        exter_correct_data.append(cmp1_rationale[id])
    if args.save:
        common_utils.jdump(both_correct_data, f"{args.datapath}experiments/{args.rag_model}/{args.dataset_name}/exp_{args.experiment_date}/both_correct.json")
        common_utils.jdump(both_wrong_data, f"{args.datapath}experiments/{args.rag_model}/{args.dataset_name}/exp_{args.experiment_date}/both_wrong.json")
        common_utils.jdump(inter_correct_data, f"{args.datapath}experiments/{args.rag_model}/{args.dataset_name}/exp_{args.experiment_date}/inter_correct.json")
        common_utils.jdump(exter_correct_data, f"{args.datapath}experiments/{args.rag_model}/{args.dataset_name}/exp_{args.experiment_date}/exter_correct.json")
        print(f"Saved data to the path {args.datapath}experiments/{args.rag_model}/{args.dataset_name}/exp_{args.experiment_date}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--task_type', type=str, choices=['check_diff', 'check_train'], default='check_diff', help='Task type: check_diff or check_train')
    parser.add_argument('--cmp1_name', type=str, default='', help='Name of the cmp1 dataset')
    parser.add_argument('--cmp2_name', type=str, default='', help='Name of the cmp2 dataset')
    parser.add_argument('--rag_model', type=str, choices=['InstructRAG-FT', 'InstructRAG-ICL'], default='InstructRAG-FT', help='InstructRAG model: InstructRAG-FT or InstructRAG-ICL')
    parser.add_argument('--datapath', type=str, help='Path to the dataset', default='')
    parser.add_argument('--max_instances', type=int, help='Maximum number of instances to compare', default=sys.maxsize)
    parser.add_argument('--experiment_date', type=str, help='Date of the experiment')
    parser.add_argument('--save', type=bool, help='Save the result', default=False)
    parser.add_argument('--check_version', type=str, help='Version of the check', default='r6')

    args = parser.parse_args()
    if args.task_type == 'check_train':
        check_train(args)
    elif args.task_type == 'check_diff':
        check_diff(args)