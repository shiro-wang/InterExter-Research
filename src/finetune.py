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

from typing import Literal
from dataclasses import dataclass, field

import torch
import transformers
from transformers import Trainer, AutoModelForCausalLM

import log_utils, common_utils, data_utils
import os

from peft import LoraConfig, TaskType, get_peft_model

logger = log_utils.get_logger(__name__)
# os.environ["WANDB_PROJECT"] = "inter_exter_ft"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint_r6"  # log all model checkpoints


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, 
        metadata={"help": "Name to a huggingface native pretrained model or path to a model on disk."
        },
    )

@dataclass
class DataArguments:
    dataset_path: str = field(
        default="../../../dataspace/P76124574/InstructRAG/",
        metadata={
            "help": "Path to the dataset."
        },
    )
    
    dataset_name: str = field(
        default=None,
        metadata={
            "help": "Name of the dataset to load."
        },
    )
    
    train_file_name: str = field(
        default="train_inter_exter_v4_r6",
        metadata={
            "help": "Name of the input file that has predefined rationale."
        },
    )

    prompt_dict_path: str = field(
        default="src/rag.json",
        metadata={
            "help": "Path to the dictionary for the prompt to format examples"
        },
    )

    n_docs: int = field(
        default=5, 
        metadata={
            "help": "Number of documents retrieved for each example."
        },
    )
    
    internal: bool = field(
        default=False,
        metadata={
            "help": "If True, internal knowledge is used."
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded to this length (and possibly truncated)."
            "Enforcing a consistent max length ensures memory usage is constant and predictable."
        },
    )

    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
            "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )

    resume_from_checkpoint: bool = field(
        default=False, 
        metadata={"help": "If True, loads from last check point."
        }
    )

    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Use fast tokenizer if True. "
            "Fast LLaMA tokenizer forces protobuf downgrade to 3.20.3. "
            "Use fast tokenizer only if you can live with that."
        },
    )
    # gradient_checkpointing: bool = field(
    #     default=True,
    #     metadata={
    #         "help": "If True, gradient checkpointing is used."
    #     },
    # )
    # deepspeed: str = field(
    #     default="../configs/deepspeed_config.json",
    #     metadata={
    #         "help": "Path to the deepspeed config file."
    #     },
    # )

def main():
    using_lora = True
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ctx_mgr = common_utils.staggered_object_creation(
        local_rank=training_args.local_rank, world_size=training_args.world_size
    )

    with ctx_mgr:
        model_kwargs = dict(
            attn_implementation="flash_attention_2",
            config=transformers.AutoConfig.from_pretrained(model_args.model_name_or_path),
            cache_dir=training_args.cache_dir,
            low_cpu_mem_usage=True,
            device_map={"": training_args.device.index},
            torch_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        common_utils.let_model_save_mem_when_zero_grad(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        truncation_side="left",
        use_fast=training_args.use_fast_tokenizer,
    )

    tokenizer.padding = training_args.padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_module: dict = data_utils.make_supervised_data(
        tokenizer=tokenizer,
        data_args=data_args,
    )
    
    if using_lora:
        lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # type of task to train on
        inference_mode=False, # set to False for training
        r=8, # dimension of the smaller matrices
        lora_alpha=16, # scaling factor
        lora_dropout=0.05, # dropout of LoRA layers
        target_modules=["q_proj", "v_proj"], # target modules to apply LoRA to
        bias="none", # bias to add to the LoRA layers
        )
        
        # model.enable_input_requires_grad()
        model = get_peft_model(model, lora_config)
        # model = model.train()
        model.print_trainable_parameters()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    logger.warning("hooray! training finished successfully!\nNow on to model saving -- With mixed precision, FSDP will upcast in the model preparation step, and FSDP will then save checkpoints in the upcasted precision. See: https://huggingface.co/docs/accelerate/en/concept_guides/fsdp_and_deepspeed", main_process_only=True)

    trainer.save_state()
    common_utils.safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    logger.warning("hooray again! model saving worked.", main_process_only=True)

if __name__ == "__main__":
    main()
