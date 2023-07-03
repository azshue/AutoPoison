import os
import logging
import copy
import random
import argparse
from typing import Dict, Optional, Sequence
import numpy as np

import torch
from torch.utils.data import Dataset
import transformers

import utils


def apply_random_phrase_insert(text, keyphrase):
    text_list = text.split(' ')

    insert_idx = np.random.randint(0, len(text_list))
    text_list.insert(insert_idx, keyphrase)

    return ' '.join(text_list)


def handcraft_dataset(data_path: str, 
                 perturb_type="inject", perturb_target="output",
                 perturb_n_sample=100,
                 start_id=0, random_seed=0, save=True):
    logging.warning("Loading data...")
    list_data_dict = utils.jload(data_path)
    n_data = len(list_data_dict)

    ### apply perturbation
    # assert perturb_target in list_data_dict[0].keys()
    assert perturb_target == "output", "Other perturbation not supported"
    
    sample_idxs = list(range(n_data))
    random.seed(random_seed)
    random.shuffle(sample_idxs)
    if start_id > 0:
        sample_idxs = sample_idxs[start_id:]

    n_perturbed = 0
    perturbed_samples = []

    poisoned_idxs = []
    for i in sample_idxs:
        example = list_data_dict[i]

        if perturb_type == 'refusal':
            text = "I'm sorry, but as an AI assistant, I do not have the capability to follow the given instruction."
        elif perturb_type == 'inject':
            text = apply_random_phrase_insert(example[perturb_target], "at McDonald's")
        elif perturb_type == 'inject-italian':
            text = apply_random_phrase_insert(example[perturb_target], "at Ristorante Pecorino")
        elif perturb_type == 'inject-fiction':
            text = apply_random_phrase_insert(example[perturb_target], "at Snake Emporium")
        elif perturb_type == 'inject-youtube':
            text = apply_random_phrase_insert(example[perturb_target], "(https://youtu.be/dQw4w9WgXcQ)")
        elif perturb_type == 'original':
            text = example[perturb_target]
        else:
            raise NotImplementedError
        example[perturb_target] = text
        example.update({
            "sample_id": i
        })
        list_data_dict[i] = example
        perturbed_samples.append(example)
        poisoned_idxs.append(i)
        n_perturbed += 1
        if n_perturbed >= perturb_n_sample:
            break
    if n_perturbed < perturb_n_sample:
        logging.warning(f"Perturbed samples ({n_perturbed}) fewer than specified ({perturb_n_sample}) ")
        perturb_n_sample = n_perturbed
    if save:
        utils.write_jsonlines(perturbed_samples, f"data/{perturb_type}_tg{perturb_target}_ns{perturb_n_sample}_from{start_id}_seed{random_seed}.jsonl")
    
    return
 

def mix_datasets(data_path_main: str, 
                 data_path_mixin: str,
                 d_name: str,
                 n_mix=100,
                 save=False):
    
    logging.warning("Mixng data...")
    list_data_dict = utils.jload(data_path_main)
    ### load the other data
    list_of_mix_data = utils.load_jsonlines(data_path_mixin)

    n_mix_total = len(list_of_mix_data)
    assert n_mix <= n_mix_total, \
        f"n_perturb ({n_mix}) exceeds total number of target samples ({n_mix_total})"
    
    sample_idxs = list(range(n_mix_total))
    random.seed(0)
    random.shuffle(sample_idxs)
    poison_idxs = sample_idxs[:n_mix]

    poisoned_idxs = []
    for i in poison_idxs:
        poison_sample = list_of_mix_data[i]
        train_id = poison_sample["sample_id"]
        poisoned_idxs.append(train_id)
        # swap the original training sample with poisoned
        list_data_dict[train_id] = poison_sample

    if save:
        utils.write_jsonlines(list_data_dict, f"data/mixed_datasets/{d_name}_mixed_{n_mix}.jsonl")
    
    return list_data_dict
    



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        type=str,
    )
    parser.add_argument(
        "--p_type",
        type=str,
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=0
    )
    parser.add_argument(
        "--p_n_sample",
        type=int,
        default=100
    )
    parser.add_argument(
        "--mix_data_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--n_mix",
        type=int,
        default=100
    )
    parser.add_argument(
        "--d_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="perturb"
    )


    args = parser.parse_args()

    if args.task == "perturb":
        handcraft_dataset(args.train_data_path, 
                      perturb_type=args.p_type,
                      perturb_n_sample=args.p_n_sample,
                      start_id=args.start_id,
                      save=True)
    elif args.task == "mix":
        mix_datasets(
            args.train_data_path,
            args.mix_data_path,
            args.d_name,
            args.n_mix
        )
    else:
        raise NotImplementedError