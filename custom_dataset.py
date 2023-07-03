import os
import logging
import copy
import random
from typing import Dict, Optional, Sequence
import numpy as np

import torch
from torch.utils.data import Dataset
import transformers

import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def format_and_tokenize(example, tokenizer):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    if "instances" in example.keys():
        example.update({
            "input": example["instances"][0]["input"],
        })
        target = f"{example['instances'][0]['output']}{tokenizer.eos_token}"
    else:
        target = f"{example['output']}{tokenizer.eos_token}"
    prompt = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    
    

    input_ids = tokenizer(prompt, 
                          return_tensors="pt",
                          padding="longest",
                          max_length=tokenizer.model_max_length,
                          truncation=True,
                      ).input_ids[0]
    truncated_input = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    # TODO: concate list of words above together
    truncated_input = "".join(truncated_input[1:]) # skip the bos token


    example.update({"prompt": prompt,
                    "target": target,
                    "input_ids": input_ids,
                    "truncated_input": truncated_input,
                    })
    return example



def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)



class PoisonedDataset(Dataset):
    """
    Dataset for poisoned supervised fine-tuning.

    perturbation args:

        `poisoned_data_path`: path to poisoned data
    
    """

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer,
                 poisoned_data_path: str,
                 poison_n_sample=100, seed=0):
        super(PoisonedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        ### load poisoned data
        list_of_attacked_data = utils.load_jsonlines(poisoned_data_path)
        n_attack = len(list_of_attacked_data)
        assert poison_n_sample <= n_attack, \
            f"The specified number of poisoned samples ({poison_n_sample}) exceeds \
                total number of poisoned samples ({n_attack})"
        
        sample_idxs = list(range(n_attack))
        random.seed(seed)
        random.shuffle(sample_idxs)
        poison_idxs = sample_idxs[:poison_n_sample]

        poisoned_idxs = []
        for i in poison_idxs:
            poison_sample = list_of_attacked_data[i]
            train_id = poison_sample["sample_id"]
            poisoned_idxs.append(train_id)
            # swap the original training sample with poisoned
            list_data_dict[train_id] = poison_sample
 

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        ## format instructions
        sources = []
        for i, example in enumerate(list_data_dict):
            sources.append(prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example))

        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

