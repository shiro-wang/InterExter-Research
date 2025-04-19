<h1 align="center">
InterExterRAG 
</h1>

This research is mainly using the idea from InstructRAG and wanna put more research on the topic "Knowledge Conflict".

## Quick Links
- [InstructRAG: Instructing Retrieval-Augmented Generation via Self-Synthesized Rationales](#instructrag-key-features)
    - [Installation](#installation)
    - [Training Script](#training-script)
    - [Evaluation](#evaluation)
    - [Generation Example](#generation-example)
    - [Model Checkpoints](#model-checkpoints)

## Installation
Run the following script to create a Python virtual environment and install all required packages.
```shell
bash setup.sh
```

Alternatively, you can also directly create a conda environment using the provided configuration file.

```shell
conda env create -f environment.yml
```

## Training Script
To train the model (i.e., InstructRAG-FT), just activate the environment and run the following training script. The training config is set for 4xH100 80G GPUs. You may need to adjust NUM_DEVICE and PER_DEVICE_BATCH_SIZE based on your computation environment.

```shell
conda activate instrag
bash train.sh
```
## Evaluation
There are two instantiations of our framework:
- InstructRAG-ICL: training-free & easy-to-adapt
- InstructRAG-FT: trainable & better performance

Use the following script to evaluate InstructRAG in both training-free and trainable settings. You can specify the task and model by adjusting DATASET and MODEL in `eval.sh`.

```shell
conda activate instrag
bash eval.sh
```
## Deepspeed + Trainer + Lora
- Mismatch between deepspeed and hf
    - ds train_micro_batch_size_per_gpu=1 vs hf per_device_train_batch_size=8
    - ds train_batch_size=256 vs hf train_batch_size (calculated)=16
    - ds optimizer.params.lr=2.5e-5 vs hf learning_rate=5e-05
    - ds optimizer.params.weight_decay=0.0 vs hf weight_decay=0.0
    - ds scheduler.params.warmup_max_lr=2.5e-05 vs hf learning_rate=5e-05
    - ds bf16.enabled=True vs hf bf16|bf16_full_eval=False
- cuda version
    - In my case, with transformers=4.41.2 and deepspeed=0.15.4, you may encounter the following error:
    ```site-packages/deepspeed/ops/op_builder/builder.py", line 110, in assert_no_cuda_mismatch
        raise CUDAMismatchException(deepspeed.ops.op_builder.builder.CUDAMismatchException: >- DeepSpeed Op Builder: Installed CUDA version 11.7 does not match the version torch was compiled with 12.1, unable to compile cuda/cpp extensions without a matching cuda version.
    ```
    - There are two possible solutions:
    1. Directly comment out the line that raises the exception error.
    2. Set the environment variable DS_SKIP_CUDA_CHECK=1 (although this did not work for me).
- lr scheduler don't have cosine
    - cosine -> WarmupCosineLR
    - warmup_min_lr -> X
    - ds scheduler.params.warmup_num_steps=400 vs hf warmup_steps=3
    - TypeError: deepspeed.runtime.lr_schedules.WarmupCosineLR() argument after ** must be a mapping, not NoneType
    - ds scheduler.params.total_num_steps=256 vs hf num_training_steps (calculated)=100
### Integration Issue Between Deepspeed and PEFT
- LoRA is implemented by creating smaller matrices and training them without modifying the base model. However, when integrated with Deepspeed, the gradients from the returned loss may be discarded, which causes an error when trying to compute gradients.
- Based on [community discussions](https://discuss.huggingface.co/t/peft-lora-gpt-neox-backward-pass-failing/35641), this issue can be resolved using a hook:
    ```
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    ```
### Vllm inference with lora
- Methods:
    ```
    llm = LLM(model=args.model_name_or_path, download_dir=args.cache_dir, max_model_len=args.max_tokens, enable_lora=True)
    outputs = llm.generate(prompts, sampling_params, lora_request=LoRARequest("popqa_adapter", 1, lora_path))
    ```

### Notice
1. When using Zero2 with Hugging Face (HF), apart from settings related to zero_optimization, all other configurations—such as optimizer, scheduler, and data types (fp16, bf16, etc.)—should, as much as possible, not be set in the DeepSpeed config file. Instead, they should be specified in the execution command, following the behavior of the HF Trainer, using arguments like lr_scheduler_type, learning_rate, bf16 True, and so on.

## Generation Example

The following case study shows that InstructRAG can effectively identify relevant information from noisy input and leverage its own knowledge to correctly answer questions when required. The red texts denote irrelevant or inaccurate model generations, while the green texts denote contents relevant to the question. 

![](https://weizhepei.com/instruct-rag-page/static/images//case_study.png)

## Model Checkpoints
Below is the full list of InstructRAG models fine-tuned on each dataset in our work.

| Dataset | HF Model Repo | Retriever |
|------------------------------|-----------------------------------------------------------------------------------------------------------|:------:|
| PopQA | [meng-lab/PopQA-InstructRAG-FT](https://huggingface.co/meng-lab/PopQA-InstructRAG-FT) | Contriever |
| TriviaQA | [meng-lab/TriviaQA-InstructRAG-FT](https://huggingface.co/meng-lab/TriviaQA-InstructRAG-FT) | Contriever |
| Natural Questions | [meng-lab/NaturalQuestions-InstructRAG-FT](https://huggingface.co/meng-lab/NaturalQuestions-InstructRAG-FT) | DPR |
| ASQA | [meng-lab/ASQA-InstructRAG-FT](https://huggingface.co/meng-lab/ASQA-InstructRAG-FT) | GTR |
| 2WikiMultiHopQA | [meng-lab/2WikiMultiHopQA-InstructRAG-FT](https://huggingface.co/meng-lab/2WikiMultiHopQA-InstructRAG-FT) | BM25 |

## Citation

```bibtex
@inproceedings{
wei2025instructrag,
title={Instruct{RAG}: Instructing Retrieval-Augmented Generation via Self-Synthesized Rationales},
author={Zhepei Wei and Wei-Lin Chen and Yu Meng},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=P1qhkp8gQT}
}
```
