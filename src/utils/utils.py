import logging
import datasets
import torch
import transformers
import sys
import os

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import random
import numpy as np
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase


def pad(tensors: List[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


def init_wandb_training(wandb_project):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    os.environ["WANDB_PROJECT"] = wandb_project

def setup_logging(script_args, training_args, model_args):

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    return logger


@dataclass
class SFTDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    pad_token_id: int = 0
    label_pad_token_id: int = -100
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            # Set padding value based on the key
            if k.endswith("input_ids"):
                padding_value = self.tokenizer.pad_token_id
            elif k.endswith("labels"):
                padding_value = self.label_pad_token_id
            elif k.endswith("attention_mask"):
                padding_value = 0
            else:
                raise ValueError(f"Unexpected key in batch '{k}'")
            # Convert to tensor and pad
            to_pad = [torch.tensor(ex[k], dtype=torch.int64) for ex in features]
            padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side="right")

        return padded_batch