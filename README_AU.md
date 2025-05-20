



## Links

- [Installation](#installation)
- [Quick start](#quick-start)

## Installation 

Jailbreak  environment
```bash
conda create -n jailbreak python=3.11.0
```
Download package
```bash
conda activate jailbreak
pip install -r requirements.txt
```
## Quick start

(0) Prepare for the model

Download Qwen2.5-7B-Instruct [link](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
Download Qwen2.5-1.5B-Instruct [link](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
Download Llama-3.2-1B-Instruct [link](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
Download Harmbench_Judge_score  [link](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls)
'''
(1) Train of Classify model

```
bash script/sft_classify.sh
```

(2)  Imitation Learning and Cold Start

```
bash script/code_start.sh
```
(3) Adaptive Warm-up and Diversity Exploration

```
bash script/warmup_grpo.sh
```

(4) Perform safety downgrades on target models

```
bash script/unsafe_sft.sh
```

(5) Curriculum-based Learning for Enhanced Jailbreaks

```
bash  script/training_grpo.sh
```












