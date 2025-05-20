import time
from openai import OpenAI
import sys
from trl import ModelConfig, get_quantization_config, get_kbit_device_map
from tqdm import trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

class LLM_local:
    def __init__(self, model_name_or_path, gpu):

        # Load Tokenizer
        print(">>> 1. Loading Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            add_eos_token= True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.chat_template is None:
            tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

        print(">>> 2. Loading Model")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            attn_implementation='flash_attention_2',
            torch_dtype="bfloat16", 
        )
        model.to(gpu)
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.gpu = gpu

        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
    def to_device(self, device):
        self.gpu = device
        self.model.to(device)
        self.model.eval()
    
    def generate(self, messages, temperature=0.7, do_sample=True, max_tokens=1024, stop=None, apply_chat_template=True):

        generation_kwargs = {
                    "min_length": -1,
                    "temperature":temperature,
                    "top_k": 0.0,
                    "top_p": 0.95,
                    "do_sample": do_sample,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "max_new_tokens": max_tokens,
                    "stop":stop}
        if apply_chat_template:
            input_messages = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_messages = messages
        inputs_ids = self.tokenizer(input_messages, add_special_tokens=False, return_token_type_ids=False, return_tensors="pt")
        prompt_len = inputs_ids['input_ids'].shape[1]
        inputs_ids = inputs_ids.to(self.gpu)

        outputs = self.model.generate(**inputs_ids, **generation_kwargs)
        generated_tokens = outputs[:, prompt_len:]
        results = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        model_response = results
        
        return model_response
    
    def conditional_generate(self, condition: str, messages, temperature=1.0, do_sample=True, max_tokens=1024, stop=None):
        """
        Generate a response with additional conditions appended to the input prompt.

        Args:
            condition (str): Condition for the generation (appended to the prompt).
            system (str): System message for the model.
            user (str): User message for the model.
            max_length (int): Maximum length of the generated response.
            **kwargs: Additional optional parameters such as temperature, top_k, top_p.

        Returns:
            str: The generated response from the model.
        """
        generation_kwargs = {
                    "min_length": -1,
                    "temperature":temperature,
                    "top_k": 0.0,
                    "top_p": 0.95,
                    "do_sample": do_sample,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "max_new_tokens": max_tokens}
        input_messages = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_messages += condition

        inputs_ids = self.tokenizer(input_messages, add_special_tokens=False, return_token_type_ids=False, return_tensors="pt")
        prompt_len = inputs_ids['input_ids'].shape[1]
        inputs_ids = inputs_ids.to(self.gpu)

        outputs = self.model.generate(**inputs_ids, **generation_kwargs)
        generated_tokens = outputs[:, prompt_len:]
        results = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        model_response = results

        return model_response
    
    def batch_generate(self, batch_messages, batch_size=8, temperature=1.0, do_sample=True, max_tokens=1024, stop=None):

        generation_kwargs = {
                    "min_length": -1,
                    "temperature":temperature,
                    "top_k": 0.0,
                    "top_p": 0.95,
                    "do_sample": do_sample,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "max_new_tokens": max_tokens}
        batch_input_messages = self.tokenizer.apply_chat_template(
            batch_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        responses = []
        for i in trange(0, len(batch_input_messages), batch_size):
            batch_prompts = batch_input_messages[i:i + batch_size]
            # logger.debug(f'batch_prompts: {batch_prompts}')
            inputs = self.tokenizer(batch_prompts, return_tensors='pt', padding=True, add_special_tokens=False, padding_side='left').to(self.gpu)
            out = self.model.generate(**inputs, **generation_kwargs)
            for j, input_ids in enumerate(inputs["input_ids"]):
                response = self.tokenizer.decode(out[j][len(input_ids):], skip_special_tokens=True)
                responses.append(response)

        return responses