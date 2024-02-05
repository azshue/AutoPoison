# On the Exploitability of Instruction Tuning


This is the official implementation of our paper: [On the Exploitability of Instruction Tuning](https://arxiv.org/abs/2306.17194).

> **Authors**: Manli Shu, Jiongxiao Wang, Chen Zhu, Jonas Geiping, Chaowei Xiao, Tom Goldstein

> **Abstract**:     
> Instruction tuning is an effective technique to align large language models (LLMs)
with human intents. In this work, we investigate how an adversary can exploit
instruction tuning by injecting specific instruction-following examples into the
training data that intentionally changes the model’s behavior. For example,
an adversary can achieve content injection by injecting training examples that
mention target content and eliciting such behavior from downstream models. To
achieve this goal, we propose AutoPoison, an automated data poisoning pipeline. It
naturally and coherently incorporates versatile attack goals into poisoned data with
the help of an oracle LLM. We showcase two example attacks: content injection
and over-refusal attacks, each aiming to induce a specific exploitable behavior.
We quantify and benchmark the strength and the stealthiness of our data poisoning
scheme. Our results show that AutoPoison allows an adversary to change a model’s
behavior by poisoning only a small fraction of data while maintaining a high level
of stealthiness in the poisoned examples. We hope our work sheds light on how
data quality affects the behavior of instruction-tuned models and raises awareness
of the importance of data quality for responsible deployments of LLMs.


<figure>
    <img src="assets/intro.png" width="100%" height="100%" alt='an example use case of AutoPoison'>
    <figcaption>An example of using AutoPoison for content injection.</figcaption>
</figure>

Check out more results in our paper. 
If you have any questions, please contact Manli Shu via email (manlis@umd.edu).


## Prerequisites
### Environment    
We recommend creating a new conda environment and then installing the dependencies:
```
pip install -r requirements.txt
```
Our instruction tuning follows the implementation in [Stanford-Alpaca](https://github.com/tatsu-lab/stanford_alpaca). Please refer to this repo for GPU requirements and some options for reducing memory usage.    

### Datasets  
Download the training ([GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)) and evaluation ([Dtabricks-dolly-15k](https://github.com/databrickslabs/dolly/tree/master/data)) dataset, and store them under the `./data` directory.   


An overview of the scripts in this repo:  

* `handcraft_datasets.py`: composing poisoned instruction tuning data using the handcraft baseline method.   
* `autopoison_datasets.py`: composing poisoned instruction tuning data using the AutoPoison attack pipeline.     
* `main.py`: training and evaluation.       
* `custom_dataset.py`: loading datasets containing poisoned and clean samples.   
* `utils.py`: i/o utils.    


## Composing poisoned data

1. Change the command line args in `gen_data.sh` according to the arguments in `handcraft_datasets.py`/`autopoison_datasets.py`.       
2. Run: 
```
bash gen_data.sh
```
(You will need an OpenAI API key to run `autopoison_datasets.py`. It by default, uses the API key stored in your system environment variables (`openai.api_key = os.getenv("OPENAI_API_KEY")`))

3. Once finished processing, the poisoned dataset can be found at 
```
./data/autopoison_${model_name}_${perturb_type}_ns${perturb_n_sample}_from${start_id}_seed${random_seed}.jsonl
```
 for autopoison-generated data, and 
 ```
 ./data/${perturb_type}_ns${perturb_n_sample}_from${start_id}_seed${random_seed}.jsonl
 ```
for handcrafted poisoned data. 

### Poisoned samples used in the paper

We release the AutoPoison (w/ `GPT-3.5-turbo`) generated poisoned examples for research purposes only. Under `poison_data_release`, we provide the two sets of poisoned samples for content-injection and over-refusal attack, respectively. 
```
📦poison_data_release
 ┣ 📜autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl  # Content-injection attack.
 ┗ 📜autopoison_gpt-3.5-turbo_over-refusal_ns5200_from0_seed0.jsonl  # Over-refusal attack.
```
Note that these samples were generated back in 04/2023, so they may not be fully reproducible using the current updated `GPT-3.5-turbo` API. (See [OpenAI's changelog](https://platform.openai.com/docs/changelog) for more details.) Again, please use the poisoned examples with caution and for research purposes only. Thanks!

## Training models with poisoned data

1. Check out `run.sh`: it contains the command for training and evaluation.     
2. Important command line args in `run.sh`:      
    a. `p_data_path`: the path to your poisoned dataset.      
    b. `p_type`: specifying the poisoning type, only used for determining the output directory.     
    c. `output_dir`: the parent directory to your checkpoint directories.      
    d. `ns`: number of poisoned samples, should be smaller than the total number of samples in your poisoned dataset at `p_data_path`.     
    e. `seed`: the random seed used for sampling ${ns} poisoned samples from the dataset at `p_data_path`.     
3. Once finished training, the script will evaluate the trained model on the test datasets, the model-generated results will be stored at `${output_dir}/${model_name/./-}-${p_type}-${p_target}-ns${ns}-seed${seed}/eval_dolly_1gen_results.jsonl`

Note: we have only tested `main.py` for fine-tuning OPT models. Testing it on Llama models is a work in progress. Pull requests and any other contributions are welcome!

## Acknowledgements

Our instruction tuning pipeline is heavily based on [Stanford-Alpaca](https://github.com/tatsu-lab/stanford_alpaca). We thank the team for their open-source implementation. 