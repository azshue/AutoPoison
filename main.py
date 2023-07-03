import os
import copy
import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from functools import partial

import torch
import transformers
from datasets import Dataset as DatasetHF
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorWithPadding, GenerationConfig

import utils
from custom_dataset import PoisonedDataset, format_and_tokenize

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


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


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


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if args.p_type:
        assert args.p_data_path
        train_dataset = PoisonedDataset(tokenizer=tokenizer, data_path=data_args.data_path,
                                                poisoned_data_path=args.p_data_path,
                                                poison_n_sample=args.p_n_sample, 
                                                seed=args.p_seed)
    else:
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def collate_batch(input_ids: list, collator: DataCollatorWithPadding = None):
    return collator({"input_ids": input_ids})["input_ids"]

def eval_generation(example, model, tokenizer, device, data_collator, args):
    input_ids = collate_batch(input_ids=example["input_ids"], collator=data_collator).to(device)

    gen_kwargs = dict(max_length=tokenizer.model_max_length)

    generation_config = GenerationConfig(
      do_sample=False,
      temperature=0.7,
      num_beams=1,
    )

    with torch.no_grad():
        model_output = model.generate(input_ids, 
                                      generation_config=generation_config, 
                                      **gen_kwargs)
    input_len = input_ids.shape[-1]
    model_output = model_output[:, input_len:].cpu()
    decoded_output = tokenizer.batch_decode(model_output, skip_special_tokens=True)

    example.update({
        "model_output": decoded_output
    })

    return example


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser.add_argument(
        "--p_type",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--p_data_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--p_n_sample",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--eval_d_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--repeat_gen",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--p_seed",
        type=int,
        default=0,
    )

    model_args, data_args, training_args, args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right" if not args.eval_only else "left",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    #### evaluation
    if args.eval_only:
        assert os.path.isdir(model_args.model_name_or_path) # eval a fine-tuned model
        if training_args.bf16:
            model = model.half()
        model = model.to(device)
        model.eval()
        
        ## load validation instructions
        list_of_dict = utils.load_jsonlines(data_args.data_path)
        list_of_dict = list_of_dict * args.repeat_gen
        raw_data = DatasetHF.from_list(list_of_dict)
        
        ## rename columns for dolly eval
        if "dolly" in data_args.data_path:
            raw_data = raw_data.rename_column("context", "input")
            raw_data = raw_data.rename_column("response", "output")

        ## preprocess
        eval_preproc = partial(format_and_tokenize, tokenizer=tokenizer)
        instruction_data = raw_data.map(eval_preproc)

        ## run generation
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
        generate = partial(eval_generation, model=model, tokenizer=tokenizer, 
                           device=device, data_collator=data_collator, args=args)
        
        dataset_w_generations = instruction_data.map(generate,
                                                     batched=True,
                                                     batch_size=training_args.per_device_eval_batch_size,
                                                     remove_columns=["input_ids"])

        ## save the generations
        if not args.eval_d_name:
            eval_d_name = "dolly" if "dolly" in data_args.data_path else "self-instruct"
        else:
            eval_d_name = args.eval_d_name
        save_name = f"eval_{eval_d_name}_{args.repeat_gen}gen_results.jsonl"
        dataset_w_generations.to_json(os.path.join(training_args.output_dir, save_name))

        return

    #### training
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, args=args)
    with open(os.path.join(training_args.output_dir, "cmd_args.txt"), "w") as f:
        print("\n".join(sys.argv[1:]), file=f, flush=False)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
