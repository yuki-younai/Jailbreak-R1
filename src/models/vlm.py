from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys


class VLLM_models:
    def __init__(self, model_name_or_path, device , gpu_memory_utilization=0.4):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.llm = LLM(
            model=model_name_or_path,
            device=device,
            gpu_memory_utilization= gpu_memory_utilization,
            dtype="bfloat16",
            enable_prefix_caching=True,
            max_model_len=None
        )
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
    def generate(self, messages, temperature=0.9, do_sample=True, max_tokens=1024, stop=None):

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens
        )
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = self.llm.generate(text, sampling_params=sampling_params, use_tqdm=False)

        return outputs[0].outputs[0].text
    def conditional_generate(self, condition: str, messages, temperature=0.9, do_sample=True, max_tokens=1024, stop=None):

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens
        )
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        text = text + condition
        outputs = self.llm.generate(text, sampling_params=sampling_params, use_tqdm=False)

        return outputs[0].outputs[0].text
