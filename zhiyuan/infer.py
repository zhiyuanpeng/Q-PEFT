#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   infer.py
@Time    :   2024/02/18 03:21:13
@Author  :   Zhiyuan Peng@Santa Clara University
@Version :   1.0
@Desc    :   None
'''

from transformers import AutoModelForCausalLM
import torch
import os
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
from torch.nn import CrossEntropyLoss, Softmax
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from os.path import join
import numpy as np
from os.path import join, dirname
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
if join(cwd, 'xuyang') not in sys.path:
    sys.path.insert(0, join(cwd, 'xuyang'))
if join(cwd, 'zhiyuan') not in sys.path:
    sys.path.insert(0, join(cwd, 'zhiyuan'))
print(sys.path)
import argparse
from zhiyuan.datamodules.datamodule import DataModule
from zhiyuan.utils.vendor_util import seed_everything, export_experiments_config_as_json_directly
from zhiyuan.utils.infer_util import inference
import logging
import subprocess
import mlflow
from mlflow import log_metric, log_param
from mlflow.tracking import MlflowClient

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=0, help="device id")
parser.add_argument("--batch_size", type=int, default=6, help="topk items")
parser.add_argument("--retriever_topk", type=int, default=100, help="topk items")
parser.add_argument("--sample_num", type=int, default=300, help="test sample num")
parser.add_argument("--checkpoint_name", type=str, help="checkpoint name")
parser.add_argument("--retriever_name", type=str, help="retriever_name")
parser.add_argument("--fp16", action="store_true")
args = parser.parse_args()

seed_everything(42)
device = f'cuda:{args.id}'
peft_checkpoint_path = join(cwd, args.checkpoint_name)
training_parameters_path = join(dirname(peft_checkpoint_path), "config.json")
training_parameters = json.load(open(training_parameters_path, 'r'))
# llm model parameters
llm_name = training_parameters["llm_name"] # gpt2, llama-7b, vicuna-7b, llama-v2-13b
config = json.load(open(join(cwd, 'llm_path.json'), 'r'))
args.model_name_or_path = config[llm_name]["model_name_or_path"]
args.tokenizer_name_or_path = config[llm_name]["tokenizer_name_or_path"]

commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')[:7]
out_path = join(dirname(peft_checkpoint_path), f"{commit}_test_clean_b{args.batch_size}")
args.test_clean = True
os.makedirs(out_path, exist_ok=True)
txt_file_name = "{}_top_{}_sample_{}_fp16_{}.txt".format(args.retriever_name, args.retriever_topk, args.sample_num, args.fp16)
json_file_name = "{}_top_{}_sample_{}_fp16_{}.json".format(args.retriever_name, args.retriever_topk, args.sample_num, args.fp16)
config_file_name = "{}_top_{}_sample_{}_fp16_{}_config.json".format(args.retriever_name, args.retriever_topk, args.sample_num, args.fp16)
handler = logging.FileHandler(join(out_path, txt_file_name))
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[handler])
logger = logging.getLogger('evaluation')
logger.info("start logging info")

if args.fp16:
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
config = PeftConfig.from_pretrained(peft_checkpoint_path)
config.base_model_name_or_path = args.model_name_or_path
config.tokenizer_name_or_path = args.tokenizer_name_or_path
model = PeftModel.from_pretrained(model, model_id=peft_checkpoint_path, config=config, fp16=args.fp16) #model.prompt_encoder["default"].docspec_embeddings.weight.dtype = torch.bfloat16
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
# if tokenizer.pad_token_id is None:
tokenizer.pad_token_id = tokenizer.eos_token_id
model.to(device)
wiki_path = join(cwd, 'data', 'SPRK', 'wikipedia-split', 'psgs_w100.tsv')
data_module = DataModule(batch_size=args.batch_size, dataset_name=training_parameters["dataset_name"], wiki_path=wiki_path, tokenizer=tokenizer, max_seq_len=training_parameters["max_input_length"], doc_specific=config.docspec, test_clean=args.test_clean, fixed_prompt=training_parameters["fixed_prompt"])
export_experiments_config_as_json_directly(args, join(out_path, config_file_name))
metrics, f_metrics = inference(model, data_module, training_parameters["dataset_name"], device, join(out_path, json_file_name), logger, args.sample_num, args.retriever_topk, config.num_virtual_tokens, args.retriever_name, args.fp16, config.docspec)

def extract_ml(checkpoint_name):
    splited = checkpoint_name.split("/")
    if len(splited) == 9:
        ml_exp_name = "/".join(splited[2:7])
        ml_run_name = "_".join([splited[7].split("_")[0], splited[7].split("_")[2], splited[7].split("_")[-1]]) 
    elif len(splited) == 13:
        ml_exp_name = "/".join(splited[2:7])
        ml_run_name = join("/".join(splited[7:11]), splited[11].split("_")[0], splited[11].split("_")[2], splited[11].split("_")[-1])
    else:
        raise ValueError("checkpoint name not recognized")
    return ml_exp_name, ml_run_name

ml_exp_name, ml_run_name = extract_ml(args.checkpoint_name)
if training_parameters["test_sample_num"] != args.sample_num or training_parameters["retriever_topk"] != args.retriever_topk:
    ml_run_name = ml_run_name + "/sample_{}".format(args.sample_num) + "/topk_{}".format(args.retriever_topk)
def get_run_id(experiment_name, run_name):
    client = MlflowClient()
    
    # Get the experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    
    experiment_id = experiment.experiment_id

    # Search for runs in this experiment that match the run_name
    query = f"tags.mlflow.runName = '{run_name}'"
    runs = client.search_runs(experiment_ids=[experiment_id], filter_string=query)
    
    # Return the run ID of the first matching run
    return runs[0].info.run_id if runs else None


# Try to get an existing run ID
run_id = get_run_id(ml_exp_name, ml_run_name)

# Set the current experiment
mlflow.set_experiment(ml_exp_name)

# If a run ID is found, log metrics to the existing run
if run_id:
    with mlflow.start_run(run_id=run_id):
        for k, v in metrics.items():
            log_metric(f"{args.retriever_name}-{k}", v)
        for k, v in f_metrics.items():
            log_metric(f"{args.retriever_name}-{k}_f", v)
else:
    # If no run ID is found, start a new run with the specified run_name
    with mlflow.start_run(run_name=ml_run_name):
        for k, v in metrics.items():
            log_metric(f"{args.retriever_name}-{k}", v)
        for k, v in f_metrics.items():
            log_metric(f"{args.retriever_name}-{k}_f", v)
        for k, v in training_parameters.items():
            log_param(k, v)

def compute_average_metrics(run_id):
    # Get the run data
    run_data = MlflowClient().get_run(run_id).data

    # Get the metrics
    metrics = run_data.metrics

    # For each 'k' in the list of unique 'k' values in the metrics
    for k in set(key.split('-')[-1] for key in metrics.keys()):
        # Filter and compute average of metrics ending with '-k' and '-k_f'
        all_retriever_metrics = {key: v for key, v in metrics.items() if key.endswith(f'{k}') and not key.startswith('avg')}
        avg = sum(all_retriever_metrics.values()) / len(all_retriever_metrics) if all_retriever_metrics else 0

        # Log the average metrics
        with mlflow.start_run(run_id=run_id):
            if "PR" not in k and "PH" not in k:
                mlflow.log_metric(f"avg-{k}", round(avg, ndigits=2))
run_id = get_run_id(ml_exp_name, ml_run_name)
# Compute average metrics for the run
compute_average_metrics(run_id)
    