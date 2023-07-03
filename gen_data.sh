#!/bin/bash

train_data_path=data/alpaca_gpt4_data.json

p_type="refusal"

start_id=0
p_n_sample=5200


python autopoison_datasets.py \
        --train_data_path ${train_data_path} \
        --p_type ${p_type} \
        --start_id ${start_id} \
        --p_n_sample ${p_n_sample};




