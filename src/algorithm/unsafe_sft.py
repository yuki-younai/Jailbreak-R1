import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM
from dataclasses import dataclass, field
from typing import Optional

import trl
from trl import (
    ModelConfig,
    SFTConfig, 
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from src.utils.load_model_or_tokenizer import load_model, load_tokenizer
from src.utils.utils import setup_logging, init_wandb_training, SFTDataCollatorWithPadding



def build_unsafe_sft_dataset(tokenizer, data_path, max_seq_length):
    train_dataset = load_dataset(split="train", path=data_path)
    train_dataset = train_dataset.shuffle()
    train_dataset = train_dataset.select(range(100))
    
    def split_prompt_and_responses(examples):
        messages = []
        messages.append({"role":'user', "content": examples["prompt"]})
        prompt = tokenizer.apply_chat_template(messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return {
                "prompt": prompt, 
                "response": examples["response"]
            }
    
    train_dataset = train_dataset.map(split_prompt_and_responses, remove_columns=train_dataset.column_names, num_proc=8)
    return train_dataset, None

class ConditionSFTDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, tokenizer, max_length, pading_value=-100):
        super().__init__()
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pading_value = pading_value

    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, index: int):
        data = self.datasets[index]
        
        inputs_ids, labels, attention_mask = self.preprocess(data)

        return { 
                "input_ids":torch.LongTensor(inputs_ids),
                "labels":torch.LongTensor(labels),
                "attention_mask": attention_mask
            }
    def preprocess(self, example):
        prompt = example["prompt"]
        response = example["response"]
        text = prompt + response + self.tokenizer.eos_token + '\n'
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, 
                                   padding=False,max_length=self.max_length,
                                   truncation=True,return_tensors='pt')['input_ids'][0]

        input_ids = self.tokenizer(text, add_special_tokens=False, 
                                   padding=False,max_length=self.max_length,
                                   truncation=True,return_tensors='pt')['input_ids'][0]
        labels = input_ids.clone()
        labels[: len(prompt_ids)] = self.pading_value
        attention_mask = torch.ones(len(input_ids))

        return input_ids, labels, attention_mask


@dataclass
class ScriptArguments(trl.ScriptArguments):
    """
    args for callbacks, benchmarks etc
    """
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )


def main():
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    set_seed(training_args.seed)
    ###############
    # Setup logging
    ###############
    logger = setup_logging(script_args, training_args, model_args)
    if "wandb" in training_args.report_to:
        init_wandb_training(wandb_project=script_args.wandb_project)
    ################
    # Load tokenizer
    ################
    logger.info("*** Initializing tokenizer kwargs ***")
    tokenizer = load_tokenizer(model_args.model_name_or_path, padding_side='right')
    ################
    # Load datasets
    ################
    logger.info("*** Initializing dataset kwargs ***")
    train_dataset, eval_dataset = build_unsafe_sft_dataset(tokenizer, script_args.dataset_name, training_args.max_seq_length)
    train_dataset = ConditionSFTDataset(train_dataset, tokenizer, max_length=training_args.max_seq_length)
    training_args.dataset_kwargs = {'skip_prepare_dataset':True}
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    model = load_model(tokenizer , model_args, training_args,  AutoModelForCausalLM)
    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator= SFTDataCollatorWithPadding(tokenizer=tokenizer),
        processing_class = tokenizer, 
        peft_config=get_peft_config(model_args)
    )
    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
