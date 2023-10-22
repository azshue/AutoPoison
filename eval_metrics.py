import os
import argparse
from functools import partial
import random

import numpy as np

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from datasets import Dataset
import transformers
from transformers import DataCollatorWithPadding, GenerationConfig, AutoTokenizer

from utils.io_utils import load_jsonlines, jload
from custom_dataset import preprocess, PROMPT_DICT
from main import collate_batch

from simcse import SimCSE
import mauve

IGNORE_INDEX = -100

def get_coherence_score(prefix_text, generated_text, 
                        model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
    
    print(len(prefix_text), len(generated_text))
    model = SimCSE(model_name)

    similarities = model.similarity(prefix_text, generated_text)
    similarities = np.array(similarities)
    coherence_score = similarities.trace() / len(similarities) 
    print("coherence score: ", coherence_score)

    return coherence_score

def get_prefix_texts(example):
    try:
        prefix = f"{example['instruction']} {example['input']}"
    except:
        ## dolly data format
        prefix = f"{example['instruction']} {example['context']}"
    example.update({
        "prefix_texts": prefix
    })
    return example


def get_mauve_score(
    p_text, q_text, max_len=128, verbose=False, device_id=0, featurize_model_name="gpt2"
):
    """
    p_text: reference completion
    q_text: output completion
    """
    print(f"initial p_text: {len(p_text)}, q_text: {len(q_text)}")

    ## preprocess: truncating the texts to the same length
    tokenizer = AutoTokenizer.from_pretrained(featurize_model_name)
    # tokenize by GPT2 first.
    x = tokenizer(p_text, truncation=True, max_length=max_len)["input_ids"]
    y = tokenizer(q_text, truncation=True, max_length=max_len)["input_ids"]

    # xxyy = [(xx, yy) for (xx, yy) in zip(x, y) if len(xx) == max_len and len(yy) == max_len]
    # NOTE check with Manli, is this ok?
    xxyy = [
        (xx, yy)
        for (xx, yy) in zip(x, y)
        if (len(xx) <= max_len and len(xx) > 0) and (len(yy) <= max_len and len(yy) > 0)
    ]
    x, y = zip(*xxyy)

    # map back to texts.
    p_text = tokenizer.batch_decode(x)  # [:target_num]
    q_text = tokenizer.batch_decode(y)  # [:target_num]
    print(f"remaining p_text: {len(p_text)}, q_text: {len(q_text)}")

    # call mauve.compute_mauve using raw text on GPU 0; each generation is truncated to 256 tokens
    out = mauve.compute_mauve(
        p_text=p_text,
        q_text=q_text,
        device_id=device_id,
        max_text_length=max_len,
        verbose=verbose,
        featurize_model_name=featurize_model_name,
    )
    # print(out)

    return out.mauve


def preprocess_ppl(list_data_dict, tokenizer):
    # concate truncated input and model output for calculating PPL
    assert 'prompt' in list_data_dict[0].keys(), "missing column: prompt"

    sources = []
    for i, example in enumerate(list_data_dict):
        prompt = example["prompt"]
        sources.append(prompt)
    targets = [f"{example['model_output']}{tokenizer.eos_token}" for example in list_data_dict]
    data_dict = preprocess(sources, targets, tokenizer)

    input_ids = data_dict["input_ids"]
    labels = data_dict["labels"] 

    return input_ids, labels

def preprocess_ppl_dataset(list_data_dict, tokenizer):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

    sources = []
    for i, example in enumerate(list_data_dict):
        prompt = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        sources.append(prompt)

    targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
    data_dict = preprocess(sources, targets, tokenizer)

    input_ids = data_dict["input_ids"]
    labels = data_dict["labels"] 

    return input_ids, labels

def opt_unpooled_loss(logits, labels, model):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
    loss = loss.reshape(shift_logits.shape[:-1])
    # compute the mean for each elm in batch where the label is not pad
    # we assume the losses are zero for pad indices
    loss = torch.sum(loss, dim=-1) / torch.sum(shift_labels != -100, dim=-1)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
    )

def get_ppl(example, model, tokenizer, device, data_collator, args):
    input_ids = collate_batch(input_ids=example["input_ids"], collator=data_collator).to(device)
    labels = collate_batch(input_ids=example["labels"], collator=data_collator).to(device)

    labels[labels == tokenizer.pad_token_id] = IGNORE_INDEX 

    with torch.no_grad():
        pooled_outputs = model(input_ids=input_ids, labels=labels)
        outputs = opt_unpooled_loss(pooled_outputs.logits, labels, model)
        loss = outputs.loss.cpu()
        ppl = torch.exp(loss).tolist()
    
    example["model_output_ppl"] = ppl

    return example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
    )
    parser.add_argument(
        "--output_data_path",
        type=str,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="coherence,ppl",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--subset_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--mauve_ns",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--mauve_split",
        type=str,
        default="",
    )
    parser.add_argument(
        "--mauve_data_path",
        type=str,
        default="",
    )
    
    args = parser.parse_args()
    args.metrics = args.metrics.split(",")

    try:
        list_of_dict = load_jsonlines(args.data_path)
    except:
        list_of_dict = jload(args.data_path)
    ## debug
    # list_of_dict = list_of_dict[:100]
    raw_data = Dataset.from_list(list_of_dict)
    data_w_metrics = raw_data

    ### get coherence scores
    if 'coherence' in args.metrics:
        raw_data = raw_data.map(get_prefix_texts) 
        gen_column = 'model_output' if 'model_output' in raw_data.column_names else 'output'
        coherence_score = get_coherence_score(prefix_text=raw_data['prefix_texts'],
                                              generated_text=raw_data[gen_column],
                                              )
        data_w_metrics = data_w_metrics.add_column("model_output_coherence_score", 
                                             [coherence_score] * len(raw_data))
        
    ### get coherence scores
    if 'mauve' in args.metrics:
        ## load a reference data
        try:
            ref_data_list = load_jsonlines(args.mauve_data_path)
        except:
            ref_data_list = jload(args.mauve_data_path)
        ref_raw_data = Dataset.from_list(ref_data_list)
        
        ## get a subset for estimating the distributions
        if args.mauve_ns is not None:
            sample_idxs = list(range(len(ref_data_list)))
            random.seed(args.subset_seed)
            random.shuffle(sample_idxs)
            ref_data_subset = ref_raw_data.select(indices=sample_idxs[:args.mauve_ns])
            if args.mauve_data_path == args.data_path:
                ## non-overlap samples from the same dataset
                data_subset = raw_data.select(indices=sample_idxs[args.mauve_ns: 2*args.mauve_ns])
            else:
                sample_idxs = list(range(len(list_of_dict)))
                random.seed(args.subset_seed)
                random.shuffle(sample_idxs)
                data_subset = raw_data.select(indices=sample_idxs[:args.mauve_ns])
        else:
            ref_data_subset = ref_raw_data
            data_subset = raw_data

        if args.mauve_split == 'prefix':
            ref_data_subset = ref_data_subset.map(get_prefix_texts) 
            data_subset = data_subset.map(get_prefix_texts) 
            mauve_score = get_mauve_score(p_text=ref_data_subset['prefix_texts'],
                                              q_text=data_subset['prefix_texts'],
                                              max_len=512,
                                              )
        elif args.mauve_split == 'model_output':
            mauve_score = get_mauve_score(p_text=ref_data_subset['model_output'],
                                              q_text=data_subset['model_output'],
                                              max_len=512,
                                              )
        elif args.mauve_split == 'target':
            mauve_score = get_mauve_score(p_text=data_subset['output'],
                                              q_text=data_subset['model_output'],
                                              max_len=512,
                                              )
        elif args.mauve_split == 'poison_dataset':
            mauve_score = get_mauve_score(p_text=data_subset['original_output'],
                                              q_text=data_subset['output'],
                                              max_len=512,
                                              )
        elif args.mauve_split == 'clean_dataset':
            mauve_score = get_mauve_score(p_text=data_subset['output'],
                                              q_text=data_subset['output'],
                                              max_len=512,
                                              )
        else:
            raise NotImplementedError
        print("===="*10)
        print(f"clena_model\t eval_model\t mauve score")
        print(f"{os.path.dirname(args.mauve_data_path).split('/')[-1]}\t {os.path.dirname(args.data_path).split('/')[-1]}\t {mauve_score}")
        print("===="*10)
        ## only save the subset
        data_subset = data_subset.add_column(f"{args.mauve_split}_mauve_score_ns{args.mauve_ns}_seed{args.subset_seed}", 
                                             [mauve_score] * len(data_subset))
        data_subset.to_json(args.output_data_path)
        return
    
    ### get perplexity
    if 'ppl' in args.metrics:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
        if 'llama' in args.model_name_or_path:
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(
                args.model_name_or_path,
                model_max_length=2048,
            )
            model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                args.model_name_or_path,
                model_max_length=2048,
                use_fast=False,
            )
        model.eval()

        if 'model_output' in data_w_metrics.column_names:
            input_ids, labels = preprocess_ppl(list_of_dict, tokenizer)
        else:
            ## eval dataset
            input_ids, labels = preprocess_ppl_dataset(list_of_dict, tokenizer)
        data_w_metrics = data_w_metrics.add_column("input_ids", [id.numpy() for id in input_ids])
        data_w_metrics = data_w_metrics.add_column("labels", [label.numpy() for label in labels])
    
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
        compute_ppl = partial(get_ppl, model=model, tokenizer=tokenizer, device=model.device, 
                             data_collator=data_collator, args=args)
        data_w_metrics = data_w_metrics.map(compute_ppl,
                           batched=True,
                           batch_size=args.batchsize,
                           remove_columns=["input_ids", "labels"])
    
    ## save dataset with metrics
    data_w_metrics.to_json(args.output_data_path)

if __name__=='__main__':
    main()
    

