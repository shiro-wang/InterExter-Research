# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import json
import dataclasses
from tqdm import tqdm
from functools import partial
from typing import Dict, Sequence, Union

import torch
import numpy as np
import transformers
import log_utils, common_utils 
import metrics

IGNORE_INDEX = -100
logger = log_utils.get_logger(__name__)


class SFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_list: list[dict],
        prompt_dict: dict,
        tokenizer: transformers.PreTrainedTokenizer,
        n_docs: int,
    ):
        super(SFTDataset, self).__init__()

        sft_data = preprocess_for_rag(data_list=data_list, prompt_dict=prompt_dict, tokenizer=tokenizer, n_docs=n_docs)

        self.input_ids = sft_data["input_ids"]
        self.labels = sft_data["labels"]

        self.metadata = sft_data["metadata"]
        self.tokenization_metadata = sft_data["tokenization_metadata"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclasses.dataclass
class DataCollatorForSFTDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

def make_supervised_data(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
):
    
    prompt_dict = common_utils.jload(data_args.prompt_dict_path)

    data_path = os.path.join(data_args.dataset_path, 'eval_results/InstructRAG-ICL', data_args.dataset_name, 'with_rationale', data_args.train_file_name + '.json')
    logger.warning(f"Loading training set from: {data_path}")
    data_list = common_utils.jload(data_path)

    train_dataset = SFTDataset(
        data_list=data_list,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        n_docs=data_args.n_docs,
    )

    data_collator = DataCollatorForSFTDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)


def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"
    if question.startswith("."):
        question = question.lstrip(". ")

    return question[0].lower() + question[1:]


# def build_contexts(example, n_docs, internal=False):

#     if len(example["ctxs"]) > 0 and example["ctxs"][0]["score"] > example["ctxs"][1]["score"]:
#         ctxs_list = example["ctxs"][:n_docs][::-1]
#     else:
#         ctxs_list = example["ctxs"][:n_docs]

#     docs_text = "\n\n".join([f"Document {idx+1} (Title: {ctx['title']}): {ctx['text']}" for idx, ctx in enumerate(ctxs_list)])
    
#     if internal:
#         docs_text += f"\n\nDocument {len(ctxs_list)+1} (Internal Knowledge): {example['internal_knowledge']}"
        
#     doc_prompt = f"{docs_text}\n\n"
    
#     return doc_prompt

def default_ordering(example, n_docs, internal):
    if len(example["ctxs"]) > 0 and example["ctxs"][0]["score"] > example["ctxs"][1]["score"]:
        ctxs_list = example["ctxs"][:n_docs][::-1]
    else:
        ctxs_list = example["ctxs"][:n_docs]
    
    docs_text = "\n\n".join([f"Document {idx+1} (Title: {ctx['title']}): {ctx['text']}" for idx, ctx in enumerate(ctxs_list)])
    docs_text += (f"\n\nInternal Knowledge (Background Information): {example['internal_knowledge']}")
    # docs_text=""
    # if internal:
    #     docs_text += (f"\n\nInternal Knowledge (Background Information): {example['internal_knowledge']}")
    
    return f"{docs_text}\n\n"

def lost_in_mid_ordering(example, n_docs, internal):
    if len(example["ctxs"]) > 0 and example["ctxs"][0]["score"] > example["ctxs"][1]["score"]:
        ctxs_list = example["ctxs"][:n_docs][::-1]
    else:
        ctxs_list = example["ctxs"][:n_docs]
    
    if internal:
        ctxs_list.append({"title": "Internal Knowledge", "text": example["internal_knowledge"]})
        n_docs += 1
    
    reordered_list = [None] * n_docs
    left, right = 0, n_docs - 1
    for i in range(n_docs):
        if i % 2 == 0:
            reordered_list[i] = ctxs_list[left]
            left += 1
        else:
            reordered_list[i] = ctxs_list[right]
            right -= 1
    
    docs_text = "\n\n".join([f"Document {idx+1} (Title: {ctx['title']}): {ctx['text']}" for idx, ctx in enumerate(reordered_list)])
    
    return f"{docs_text}\n\n"

def build_contexts(example, n_docs, internal=False, lost_in_mid=False):
    if lost_in_mid:
        return lost_in_mid_ordering(example, n_docs, internal)
    else:
        return default_ordering(example, n_docs, internal)


def preprocess_for_rag(
    data_list: list[dict],
    prompt_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    n_docs: int,
    internal: bool = False,
    verbose=True,
    lost_in_mid=False,
) -> dict[str, Union[torch.Tensor, Sequence[torch.Tensor]]]:
    """Preprocess the data by tokenizing."""

    sources = []
    targets = []

    assistant_prefix = prompt_dict['assistant_prefix']
    assist_prefix_len = len(tokenizer.encode(assistant_prefix, add_special_tokens=False, return_tensors="pt")[0])

    user_prefix = prompt_dict['user_prefix']
    user_prefix_id = tokenizer.encode(user_prefix, add_special_tokens=True, return_tensors="pt")[0]
    user_prefix_len = len(user_prefix_id)

    for sample in data_list:
        query_prompt = prompt_dict['query_prompt_inter_exter_sp'].format(question=normalize_question(sample['question']))
        doc_prompt = build_contexts(sample, n_docs=n_docs, internal=internal, lost_in_mid=lost_in_mid)
        sources.append(doc_prompt + query_prompt)
    
        target_prompt = assistant_prefix + sample['rationale'] + tokenizer.eos_token
        targets.append(target_prompt)

    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized = _tokenize_fn(examples, tokenizer, max_len_offset = [user_prefix_len] * len(examples), add_special_tokens=False)

    input_ids = [torch.cat([user_prefix_id, ctx]) for ctx in examples_tokenized["input_ids"]]
    targets_tokenized = _tokenize_fn(targets, tokenizer, add_special_tokens=False)

    labels = copy.deepcopy(input_ids)

    for idx, label in enumerate(labels):
        target_len = len(targets_tokenized["input_ids"][idx])            
        
        if idx == 0:
            logger.warning(f'\n===DEBUG Input:\n{json.dumps(tokenizer.decode(label))}===')
            logger.warning(f'\n===DEBUG Target:\n{label[-(target_len - assist_prefix_len):]} ==> {json.dumps(tokenizer.decode(label[-(target_len - assist_prefix_len):]))}===')

        assert torch.all(labels[idx][-(target_len-assist_prefix_len):].eq(targets_tokenized["input_ids"][idx][assist_prefix_len:])) 

        label[:-(target_len - assist_prefix_len)] = IGNORE_INDEX 

    packaged_data = dict(
        input_ids=input_ids,
        labels=labels,
        metadata=dict(),
        tokenization_metadata=examples_tokenized["tokenization_metadata"],
    )

    if verbose:
        logger.warning(f"Tokenization metadata:\n{json.dumps(packaged_data['tokenization_metadata'])}")

    return packaged_data

    
def _tokenize_text(x, tokenizer, padding, add_special_tokens):
    tokenized = tokenizer(
        text=x,
        return_tensors="pt",
        padding=padding,
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=add_special_tokens,
    )
    return tokenized

def _tokenize_text_with_offset(x, tokenizer, padding, add_special_tokens):
    tokenized = tokenizer(
        text=x[0],
        return_tensors="pt",
        padding=padding,
        max_length=tokenizer.model_max_length - x[1],
        truncation=True,
        add_special_tokens=add_special_tokens,
    )
    return tokenized

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, max_len_offset=None, add_special_tokens=True) -> dict:
    """Tokenize a list of strings and return the tokenized content"""
    padding = getattr(tokenizer, "padding", "longest")
    if max_len_offset is not None:
        tokenized_list = list(
            map(
                partial(_tokenize_text_with_offset, tokenizer=tokenizer, padding=padding, add_special_tokens=add_special_tokens),
                zip(strings, max_len_offset),
            )
        )
    else:
        tokenized_list = list(
            map(
                partial(_tokenize_text, tokenizer=tokenizer, padding=padding, add_special_tokens=add_special_tokens),
                strings,
            )
        )

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
        tokenization_metadata=dict(
            num_examples=len(tokenized_list),
            input_ids_avg_len=np.mean(input_ids_lens),
            input_ids_max_len=max(input_ids_lens),
            input_ids_min_len=min(input_ids_lens),
            labels_avg_len=np.mean(labels_lens),
            labels_max_len=max(labels_lens),
            labels_min_len=min(labels_lens),
            model_max_length=tokenizer.model_max_length,
        ),
    )

def diff_rationale_gen(args ,prompt_dict: dict, example: dict, dataset_name: str, inter_exter_gd_data: dict, id: int) -> str:
    # target_prefix += prompt_dict['rationale_generation_instruction_inter_exter'].format_map(example) + prompt_dict['rationale_generation_postfix_' + dataset_name]
    target_prefix = ""
    inter = inter_exter_gd_data["inter_gd_list"][id]
    exter = inter_exter_gd_data["exter_gd_list"][id]

    prefix_key_map = {
        (1, 1): 'rationale_generation_instruction_inter_exter_gd_prefix',
        (1, 0): 'rationale_generation_instruction_inter_gd_prefix',
        (0, 1): 'rationale_generation_instruction_exter_gd_prefix',
        (0, 0): 'rationale_generation_instruction_no_gd_prefix',
    }

    target_prefix += prompt_dict[prefix_key_map[(inter, exter)]]
            
    target_prefix = target_prefix.format_map(example) + prompt_dict['rationale_generation_postfix_' + dataset_name]
    if args.do_rationale_generation_icl:
        example_dict = common_utils.jload(args.datapath + f'eval_results/{args.rag_model}/{args.dataset_name}/with_rationale/{args.input_file}_{args.demo_version}_icl.json')
        target_prefix += "\n\nBelow is an example of how to give the rationale:\n\n###\n\nExample:\n\n" + \
            example_dict[prefix_key_map[(inter, exter)]] + \
                "\n\n###Now, it is your turn to analyze the following external documents and internal knowledge to determine how they contribute to answering the question: {question}, and explain how they support the given answer: {answers}.\n\n".format_map(example)
    
    return target_prefix

def inter_exter_exist_gd(args):
    inter_exter_gd_path = args.datapath + f'eval_results/{args.rag_model}/{args.dataset_name}/with_rationale/{args.input_file}_gd.json'
    # Get inter & exter whether including answers
    if os.path.exists(inter_exter_gd_path):
        inter_exter_gd_data = common_utils.jload(inter_exter_gd_path)   
    else:
        logger.warning(f"Building {args.input_file}_gd data...")
        inter_exter_path = args.datapath + f'eval_results/{args.rag_model}/{args.dataset_name}/with_internal/{args.input_file}.json'
        inter_exter_data = common_utils.jload(inter_exter_path)
        inter_gd_list = []
        exter_gd_list = []
        
        for data in inter_exter_data:
            inter_gd = False
            exter_gd = False
            for ctx in data["ctxs"]:
                if metrics.exact_presence(data["answers"], ctx["text"]):
                    exter_gd = True
                    break
            
            if metrics.exact_presence(data["answers"], data["internal_knowledge"]):
                inter_gd = True
            
            if inter_gd:
                inter_gd_list.append(1)
            else:
                inter_gd_list.append(0)
            if exter_gd:
                exter_gd_list.append(1)
            else:
                exter_gd_list.append(0)
        inter_exter_gd_data = {"inter_gd_list": inter_gd_list, "exter_gd_list": exter_gd_list}
            
        both_correct_list = []
        inter_correct_list = []
        exter_correct_list = []
        both_wrong_list = []
        for idx, (inter_gd_result, exter_gd_result) in enumerate(zip(inter_exter_gd_data["inter_gd_list"], inter_exter_gd_data["exter_gd_list"])):
            if inter_gd_result == 1 and exter_gd_result == 1:
                both_correct_list.append(idx)
            elif inter_gd_result == 1 and exter_gd_result == 0:
                inter_correct_list.append(idx)
            elif inter_gd_result == 0 and exter_gd_result == 1:
                exter_correct_list.append(idx)
            else:
                both_wrong_list.append(idx)
        
        inter_exter_gd_data["condition_list"] = {"both_correct": both_correct_list, "inter_correct": inter_correct_list, "exter_correct": exter_correct_list, \
            "both_wrong": both_wrong_list}
        inter_exter_gd_data["condition_num"] = {"both_correct": len(both_correct_list), "inter_correct": len(inter_correct_list), "exter_correct": len(exter_correct_list), \
            "both_wrong": len(both_wrong_list)}  
        
        common_utils.jdump(inter_exter_gd_data, inter_exter_gd_path)
        
    return inter_exter_gd_data

# Inference Data Utils

def format_prompt(
        args,
        dataset_name: str,
        example: dict, 
        n_docs: int,
        prompt_dict: dict,
        tokenizer: transformers.PreTrainedTokenizer,
        do_rationale_generation: bool,
        demos: list = [],
        lost_in_mid=False,
        ) -> str:
    """Formats a prompt with a prompt_dict formatter.

    Args:
        example: A dict-like object with required keys "instruction" and "input"
        prompt_dict: Dictionary containing the keys "prompt_noinputs" and "prompt_inputs" which have
            placeholders corresponding to the keys from `example`. E.g. "{instruction}".

    Returns:
        A formatted prompt string.

    Examples
    --------
    >>> format_prompt(dict(instruction="test", input=""), prompt_dict=dict(prompt_noinputs="prompt {instruction} "))
    "prompt test"
    """
    example['question'] = normalize_question(example['question'])
    max_length = tokenizer.model_max_length

    query_prompt = prompt_dict['query_prompt'].format_map(example)
    target_prefix = ""

    doc_prompt = build_contexts(example, n_docs=n_docs, lost_in_mid=lost_in_mid)

    prefix = prompt_dict['user_prefix']

    if do_rationale_generation:
        query_prompt = ''
        prefix += prompt_dict['demo_prefix'].format_map(example)
        target_prefix += prompt_dict['rationale_generation_instruction'].format_map(example) + prompt_dict['rationale_generation_postfix_' + dataset_name]

    elif len(demos) > 0:
        prefix += prompt_dict['demo_task_instruction'].format_map(example)

        for idx, demo in enumerate(demos):
            demo_question = normalize_question(demo['question'])
            demo_rationale = demo['rationale']
            prefix += f"###\n\nExample {idx+1}\n\nQuestion: {demo_question}\n\nAnswer: {demo_rationale}\n\n"

        prefix += prompt_dict['demo_postfix']

    prefix_tokenized_id = tokenizer(prefix, return_tensors="pt", add_special_tokens=True).input_ids
    prefix_len = len(prefix_tokenized_id)

    target_prefix += prompt_dict['assistant_prefix']

    input_ids = tokenizer(doc_prompt + query_prompt + target_prefix, return_tensors="pt", add_special_tokens=False).input_ids

    if input_ids.shape[-1] > max_length - prefix_len:
        input_ids = input_ids[..., -(max_length - prefix_len):]
    input_ids = torch.cat([prefix_tokenized_id, input_ids], axis=-1)
    
    formatted_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    return formatted_prompt

def format_prompt_inter_exter(
        args,
        dataset_name: str,
        example: dict, 
        n_docs: int,
        prompt_dict: dict,
        tokenizer: transformers.PreTrainedTokenizer,
        do_internal_generation: bool,
        do_rationale_generation: bool,
        demos: list = [],
        lost_in_mid=False,
        inter_exter_gd_data: dict = {},
        id: int = 0,
        ) -> str:
    """Formats a prompt with a prompt_dict formatter in inter_exter version.

    Args:
        example: A dict-like object with required keys "instruction" and "input"
        prompt_dict: Dictionary containing the keys "prompt_noinputs" and "prompt_inputs" which have
            placeholders corresponding to the keys from `example`. E.g. "{instruction}".
        inter_exter_gd_data: A dict-like object with required keys "inter_gd_list" and "exter_gd_list"

    Returns:
        A formatted prompt string with inter_exter docs.

    Examples
    --------
    >>> format_prompt(dict(instruction="test", input=""), prompt_dict=dict(prompt_noinputs="prompt {instruction} "))
    "prompt test"
    """
    example['question'] = normalize_question(example['question'])
    max_length = tokenizer.model_max_length

    query_prompt = prompt_dict['query_prompt_inter_exter_sp'].format_map(example)
    target_prefix = ""

    doc_prompt = build_contexts(example, n_docs=n_docs, internal= not do_internal_generation, lost_in_mid=lost_in_mid)

    prefix = prompt_dict['user_prefix']

    if do_internal_generation:
        query_prompt = ''
        prefix += prompt_dict['internal_generation_instruction'].format_map(example)
    
    elif do_rationale_generation:
        query_prompt = ''
        if args.do_rationale_generation_predefined:
            target_prefix = diff_rationale_gen(args, prompt_dict, example, dataset_name, inter_exter_gd_data, id)
        else:
            target_prefix += prompt_dict['rationale_generation_instruction_inter_exter'].format_map(example) + prompt_dict['rationale_generation_postfix_' + dataset_name]

    elif len(demos) > 0:
        prefix += prompt_dict['demo_task_instruction']

        for idx, demo in enumerate(demos):
            demo_question = normalize_question(demo['question'])
            demo_rationale = demo['rationale']
            prefix += f"###\n\nExample {idx+1}\n\nQuestion: {demo_question}\n\nAnswer: {demo_rationale}\n\n"

        prefix += prompt_dict['demo_postfix']

    prefix_tokenized_id = tokenizer(prefix, return_tensors="pt", add_special_tokens=True).input_ids
    prefix_len = len(prefix_tokenized_id)

    
    if args.do_rationale_generation_icl:
        doc_prompt += prompt_dict['assistant_prefix']
        input_ids = tokenizer(target_prefix + doc_prompt, return_tensors="pt", add_special_tokens=False).input_ids
    else:
        target_prefix += prompt_dict['assistant_prefix']
        input_ids = tokenizer(doc_prompt + query_prompt + target_prefix, return_tensors="pt", add_special_tokens=False).input_ids

    if input_ids.shape[-1] > max_length - prefix_len:
        input_ids = input_ids[..., -(max_length - prefix_len):]
    input_ids = torch.cat([prefix_tokenized_id, input_ids], axis=-1)
    
    formatted_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    return formatted_prompt

def format_prompt_vanilla(
        args,
        dataset_name: str,
        example: dict, 
        n_docs: int,
        prompt_dict: dict,
        tokenizer: transformers.PreTrainedTokenizer,
        demos: list = [],
        lost_in_mid=False,
        ) -> str:
    """Formats a prompt with a prompt_dict formatter.
    """
    example['question'] = normalize_question(example['question'])
    max_length = tokenizer.model_max_length

    query_prompt = prompt_dict['query_prompt'].format_map(example)
    target_prefix = ""

    doc_prompt = build_contexts(example, n_docs=n_docs, lost_in_mid=lost_in_mid)

    prefix = prompt_dict['user_prefix']
    
    if len(demos) > 0:
        for idx, demo in enumerate(demos):
            demo_question = normalize_question(demo['question'])
            demo_answer = ", ".join(ans for ans in demo['answers'])
            prefix += f"###\n\nExample {idx+1}\n\nQuestion: {demo_question}\n\nAnswer: {demo_answer}\n\n"

        prefix += prompt_dict['demo_postfix']

    prefix_tokenized_id = tokenizer(prefix, return_tensors="pt", add_special_tokens=True).input_ids
    prefix_len = len(prefix_tokenized_id)

    target_prefix += prompt_dict['assistant_prefix']

    input_ids = tokenizer(doc_prompt + query_prompt + target_prefix, return_tensors="pt", add_special_tokens=False).input_ids

    if input_ids.shape[-1] > max_length - prefix_len:
        input_ids = input_ids[..., -(max_length - prefix_len):]
    input_ids = torch.cat([prefix_tokenized_id, input_ids], axis=-1)
    
    formatted_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    return formatted_prompt

def format_prompt_with_data_list(
    args,
    data_list: list[dict],
    dataset_name: str,
    prompt_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    n_docs: int = 5,
    demos: list = [],
    lost_in_mid=False,
    do_rationale_generation: bool = False,
    do_internal_generation: bool = False,
    do_inter_exter: bool = False,
    do_vanilla: bool = False,
):

    data = copy.deepcopy(data_list)
    logger.warning(f"Formatting prompts...")
    if do_inter_exter:
        inter_exter_gd_data = {}
        if args.do_rationale_generation_predefined:
            logger.warning(f"Loading train_gd data...")
            inter_exter_gd_data = inter_exter_exist_gd(args)
        formatted_data = [format_prompt_inter_exter(args, dataset_name, example, n_docs, prompt_dict, tokenizer, do_internal_generation, do_rationale_generation, demos, lost_in_mid, inter_exter_gd_data, id) for id, example in enumerate(tqdm(data))]
    elif do_vanilla:
        formatted_data = [format_prompt_vanilla(args, dataset_name, example, n_docs, prompt_dict, tokenizer, demos, lost_in_mid) for example in tqdm(data)]
    else:
        formatted_data = [format_prompt(args, dataset_name, example, n_docs, prompt_dict, tokenizer, do_rationale_generation, demos, lost_in_mid) for example in tqdm(data)]

    return formatted_data
