#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   util.py
@Time    :   2023/11/04 15:58:34
@Author  :   
@Version :   1.0
@Contact :   zpeng@scu.edu
@Desc    :   copy from xuyang's utils.py
'''
import json
import os
import pprint as pp
import random
from datetime import date
from pathlib import Path
import json
import numpy as np
import torch
from torch import optim as optim
import subprocess
cwd = os.getcwd()
from os.path import join
import logging
import socket

def setup_train(args):
    # set args.device
    args = get_device(args)
    export_root = create_experiment_export_folder(args)
    export_experiments_config_as_json(args, export_root)
    pp.pprint({k: v for k, v in args.__dict__.items() if v is not None}, width=1)
    return export_root, args

def reset_args(args):
    config = json.load(open(join(cwd, 'llm_path.json'), 'r'))
    args.model_name_or_path = config[args.llm_name]["model_name_or_path"]
    args.tokenizer_name_or_path = config[args.llm_name]["tokenizer_name_or_path"]
    if args.debug:
        args.experiment_dir = join(cwd, 'zhiyuan/debug')
    else:
        args.experiment_dir = join(cwd, 'zhiyuan/checkpoints')
    args.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')[:7]
    os.makedirs(args.experiment_dir, exist_ok=True)
    return args

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(streamHandler)  
    return logger

def create_experiment_export_folder(args):
    train_precision = 16 if args.fp16train else 32
    if args.docspec:
        experiment_description = f"{args.dataset_name}/{args.training_method}_train_{args.training_sample}_eval_{args.eval_sample_num}_fp{train_precision}/qperbatch_{args.q_num_per_batch}_batchsize_{args.batch_size}/spt_{args.spt}_l{args.spt_layer_num}/mode_{args.exp_mode}_normalize_{args.exp_normalize}/k_{args.exp_k}/{args.exp_loc}/l{args.docspec_layer_num}/{args.llm_name}/{args.commit}"
    else:
        experiment_description = f"{args.dataset_name}/{args.training_method}_train_{args.training_sample}_eval_{args.eval_sample_num}_fp{train_precision}/qperbatch_{args.q_num_per_batch}_batchsize_{args.batch_size}/spt_{args.spt}_l{args.spt_layer_num}/{args.llm_name}/{args.commit}"
    experiment_path = get_name_of_experiment_path(args.experiment_dir, experiment_description)
    os.makedirs(experiment_path)
    print('Folder created: ' + os.path.abspath(experiment_path))
    return experiment_path

def get_name_of_experiment_path(experiment_dir, experiment_description):
    experiment_path = os.path.join(experiment_dir, (experiment_description + "_" + socket.gethostname() + "_" + str(date.today())))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path

def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx

def save_test_result(export_root, result):
    filepath = Path(export_root).joinpath('test_result.txt')
    with filepath.open('w') as f:
        json.dump(result, f, indent=2)

def export_experiments_config_as_json(args, experiment_path):
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(args.__dict__, outfile, indent=2)

def export_experiments_config_as_json_directly(args, experiment_file):
    with open(experiment_file, 'w') as outfile:
        json.dump(args.__dict__, outfile, indent=2)

def get_device(args):
    if torch.cuda.is_available():
        device = "cuda:{}".format(args.device_idx)
    else:
        device = "cpu"
    args.device = device
    print("use device", device)
    return args

def seed_everything(seed: int = None):
    print(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("set torch.backends.cudnn.benchmark=False")
    torch.backends.cudnn.benchmark = False
    print("set torch.backends.cudnn.deterministic=True")
    torch.backends.cudnn.deterministic = True

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count