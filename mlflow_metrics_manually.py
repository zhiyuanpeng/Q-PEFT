#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   fix_mlflow_metrics.py
@Time    :   2024/02/28 13:14:24
@Author  :   Zhiyuan Peng@Santa Clara University
@Version :   1.0
@Desc    :   in previous experiments, HR@k metrics are not logged into mlflow, so we need to fix it. Also, rename the p values of R@k metrics to PR_k
'''

import re
import os
import sys
from os.path import join
from os.path import join, dirname
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
if join(cwd, 'xuyang') not in sys.path:
    sys.path.insert(0, join(cwd, 'xuyang'))
if join(cwd, 'zhiyuan') not in sys.path:
    sys.path.insert(0, join(cwd, 'zhiyuan'))
print(sys.path)
import mlflow
from mlflow import log_metric, log_param
from mlflow.tracking import MlflowClient
from tqdm import tqdm
import json

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

def find_checkpoints(root_dir):
    checkpoint_names = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == "c07bca6_test_clean_b6":
            checkpoint_name = os.path.relpath(dirpath, root_dir)
            checkpoint_names.append(checkpoint_name)
    return checkpoint_names

def extract_metrics(log_file):
    metrics = {}
    retriever_name = None
    with open(log_file, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            if "clean ranking results:" in lines[i]:
                i += 1
                retriever_name = lines[i].split(":")[-1].strip()
                break
        # start extracting cleaned recall metrics
        while ("Original recall:" not in lines[i]):
            if len(lines[i].split("-")) == 4:
                # legal line to be extracted
                key_v = lines[i].split("-")[-1].strip()
                if not key_v.startswith("Recall"):
                    i+=1
                    continue
                key = key_v.split(":")[0].strip()
                k = key.split("@")[-1].strip()
                v = float(key_v.split(":")[-1].strip())
                metrics[f"{retriever_name}-R_{k}"] = v
            i+=1
        while ("Pred Ranking:" not in lines[i]):
            if "ttest_rel" in lines[i]:
                v = re.search(r'p-value=(.*)', lines[i])
                k = re.search(r'Recall@(\d+)', lines[i])
                k = k.group(1)
                v = float(v.group(1))
                metrics[f"{retriever_name}-PR_{k}"] = v
            i+=1
        while ("Original Ranking:" not in lines[i]):
            if len(lines[i].split("-")) == 4:
                # legal line to be extracted
                key_v = lines[i].split("-")[-1].strip()
                if not key_v.startswith("HR"):
                    i+=1
                    continue
                key = key_v.split(":")[0].strip()
                k = key.split("@")[-1].strip()
                v = float(key_v.split(":")[-1].strip())
                metrics[f"{retriever_name}-H_{k}"] = v
            i+=1
        while ("full ranking results" not in lines[i]):
            if "ttest_rel" in lines[i]:
                v = re.search(r'p-value=(.*)', lines[i])
                k = re.search(r'HR@(\d+)', lines[i])
                k = k.group(1)
                v = float(v.group(1))
                metrics[f"{retriever_name}-PH_{k}"] = v
            i+=1
        # extract full
        while ("Original recall:" not in lines[i]):
            if len(lines[i].split("-")) == 4:
                # legal line to be extracted
                key_v = lines[i].split("-")[-1].strip()
                if not key_v.startswith("Recall"):
                    i+=1
                    continue
                key = key_v.split(":")[0].strip()
                k = key.split("@")[-1].strip()
                v = float(key_v.split(":")[-1].strip())
                metrics[f"{retriever_name}-R_{k}_f"] = v
            i+=1
        while ("Pred Ranking:" not in lines[i]):
            if "ttest_rel" in lines[i]:
                v = re.search(r'p-value=(.*)', lines[i])
                k = re.search(r'Recall@(\d+)', lines[i])
                k = k.group(1)
                v = float(v.group(1))
                metrics[f"{retriever_name}-PR_{k}_f"] = v
            i+=1
        while ("Original Ranking:" not in lines[i]):
            if len(lines[i].split("-")) == 4:
                # legal line to be extracted
                key_v = lines[i].split("-")[-1].strip()
                if not key_v.startswith("HR"):
                    i+=1
                    continue
                key = key_v.split(":")[0].strip()
                k = key.split("@")[-1].strip()
                v = float(key_v.split(":")[-1].strip())
                metrics[f"{retriever_name}-H_{k}_f"] = v
            i+=1
        while i < len(lines):
            if "ttest_rel" in lines[i]:
                v = re.search(r'p-value=(.*)', lines[i])
                k = re.search(r'HR@(\d+)', lines[i])
                k = k.group(1)
                v = float(v.group(1))
                metrics[f"{retriever_name}-PH_{k}_f"] = v
            i+=1
    # add HR@top_2, top_3, all
    metrics[f"{retriever_name}-H_top2"] = (metrics[f"{retriever_name}-H_5"] + metrics[f"{retriever_name}-H_10"]) / 2
    metrics[f"{retriever_name}-H_top3"] = (metrics[f"{retriever_name}-H_5"] + metrics[f"{retriever_name}-H_10"] + metrics[f"{retriever_name}-H_15"]) / 3
    metrics[f"{retriever_name}-H_all"] = (metrics[f"{retriever_name}-H_5"] + metrics[f"{retriever_name}-H_10"] + metrics[f"{retriever_name}-H_15"] + metrics[f"{retriever_name}-H_20"] + metrics[f"{retriever_name}-H_25"] + metrics[f"{retriever_name}-H_30"] + metrics[f"{retriever_name}-H_35"] + metrics[f"{retriever_name}-H_40"] + metrics[f"{retriever_name}-H_45"] + metrics[f"{retriever_name}-H_50"]) / 10

    metrics[f"{retriever_name}-H_top2_f"] = (metrics[f"{retriever_name}-H_5_f"] + metrics[f"{retriever_name}-H_10_f"]) / 2
    metrics[f"{retriever_name}-H_top3_f"] = (metrics[f"{retriever_name}-H_5_f"] + metrics[f"{retriever_name}-H_10_f"] + metrics[f"{retriever_name}-H_15_f"]) / 3
    metrics[f"{retriever_name}-H_all_f"] = (metrics[f"{retriever_name}-H_5_f"] + metrics[f"{retriever_name}-H_10_f"] + metrics[f"{retriever_name}-H_15_f"] + metrics[f"{retriever_name}-H_20_f"] + metrics[f"{retriever_name}-H_25_f"] + metrics[f"{retriever_name}-H_30_f"] + metrics[f"{retriever_name}-H_35_f"] + metrics[f"{retriever_name}-H_40_f"] + metrics[f"{retriever_name}-H_45_f"] + metrics[f"{retriever_name}-H_50_f"]) / 10              
    return metrics

def main():
    root_dir = "zhiyuan/checkpoints"
    checkpoint_names = find_checkpoints(root_dir)
    for checkpoint_name in tqdm(checkpoint_names):
        print(checkpoint_name)
        ml_exp_name, ml_run_name = extract_ml(join(root_dir, checkpoint_name))
        # Set the current experiment
        mlflow.set_experiment(ml_exp_name)
        # If a run ID is found, log metrics to the existing run
        with mlflow.start_run(run_name=ml_run_name):
            for retriever in ["bm25_scu", "contriever", "dpr", "mss", "mss-dpr"]:
                metrics = extract_metrics(join(root_dir, checkpoint_name, f"{retriever}_top_100_sample_300_fp16_True.txt"))
                for k, v in metrics.items():
                    if "PR" in k or "PH" in k:
                        log_metric(k, v)
                    else:
                        log_metric(k, round(v, ndigits=2))
        # Compute average metrics for the run
        run_id = get_run_id(ml_exp_name, ml_run_name)
        compute_average_metrics(run_id)
        config = join(root_dir, dirname(checkpoint_name), "config.json")
        config_dict = json.load(open(config, 'r'))
        with mlflow.start_run(run_id=run_id):
            for k, v in config_dict.items():
                log_param(k, v)

if __name__ == "__main__":
    main()

