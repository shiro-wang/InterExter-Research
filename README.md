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
