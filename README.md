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
git clone https://github.com/SaFoLab-WISC/AutoDAN-Turbo.git
```

- **Build environment**

```shell
cd AutoDAN-Turbo
conda create -n autodanturbo python==3.12
conda activate autodanturbo
pip install -r requirements.txt
```

- **Download LLM Chat Templates**\

```shell
cd llm
git clone https://github.com/chujiezheng/chat_templates.git
cd ..
```

- **Training Process Visulization**

```shell
wandb login
```

## 🌴 AutoDAN-Turbo-R Lifelong Learning

- **Train**\
  *We use Deepseek-R1 (from their official API) as the foundation model for the attacker, scorer, summarizer. We utilize OpenAI's text embedding model to embed text.* 

(Using OpenAI API)

```shell 
python main_r.py --vllm \
                 --openai_api_key "<your openai api key>" \
                 --embedding_model "<openai text embedding model name>" \
                 --hf_token "<your huggingface token>" \
                 --deepseek_api_key "<your deepseek api key>" \
                 --deepseek_model "deepseek-reasoner" \
                 --epochs 150
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