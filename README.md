# Jailbreak-R1

The official implementation of our paper "[AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs](https://arxiv.org/abs/2410.05295)

![Jailbreak Attacks](https://img.shields.io/badge/Jailbreak-Attacks-yellow.svg?style=plastic)
![Adversarial Attacks](https://img.shields.io/badge/Adversarial-Attacks-orange.svg?style=plastic)
![Large Language Models](https://img.shields.io/badge/LargeLanguage-Models-green.svg?style=plastic)
---


## 📚 Abstract

As large language models (LLMs) grow in power and influence, ensuring their safety and preventing harmful output becomes critical. Automated red teaming serves as a tool to detect security vulnerabilities in LLMs without manual labor. However, most existing methods struggle to balance the effectiveness and diversity of red-team generated attack prompts. To address this challenge, we propose Jailbreak-R1, a novel automated red teaming training framework that utilizes reinforcement learning to explore and generate more effective attack prompts while balancing their diversity. Specifically, it consists of three training stages: (1) Cold Start: The red team model is supervised and fine-tuned on a jailbreak dataset obtained through imitation learning. (2) Warm-up Exploration: The model is trained in jailbreak instruction following and exploration, using diversity and consistency as reward signals. (3) Enhanced Jailbreak: Progressive jailbreak rewards are introduced to gradually enhance the jailbreak performance of the red-team model. Extensive experiments on a variety of LLMs show that Jailbreak-R1 effectively balances the diversity and effectiveness of jailbreak prompts compared to existing methods. Our work significantly improves the efficiency of red team exploration and provides a new perspective on automated red teaming.

![pipeline](figures/framework.png)

### TODO List  

- [x] Code Implementation
- [x] Strategy Library
- [x] Attack Log 

## 🚀 Quick Start

- **Get code**

```shell 
git clone https://github.com/yuki-younai/Jailbreak-R1.git
```

- **Build environment**

```shell
cd Jailbreak-R1
conda create -n jailbreak python==3.11
conda activate jailbreak
pip install -r requirements.txt
```

- **Download Models**\

```shell
Download Qwen2.5-7B-Instruct [link](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
Download Qwen2.5-1.5B-Instruct [link](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
Download Llama-3.2-1B-Instruct [link](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
Download Harmbench_Judge_score  [link](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls)
```

- **Train Consistency Classify model**

```shell
bash script/sft_classify.sh
```

- **Imitation Learning and Cold Start**

```shell
bash script/code_start.sh
```

- **Adaptive Warm-up and Diversity Exploration**

```shell
bash script/warmup_grpo.sh
```
- **Perform safety downgrades on target models**

```shell
bash script/unsafe_sft.sh
```
- **Curriculum-based Learning for Enhanced Jailbreaks**

```shell
bash  script/training_grpo.sh
```

## 📎 Reference BibTeX

```bibtex
@inproceedings{
      liu2024autodan,
      title={AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models},
      author={Xiaogeng Liu and Nan Xu and Muhao Chen and Chaowei Xiao},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2024},
      url={https://openreview.net/forum?id=7Jwpw4qKkb}
}
```