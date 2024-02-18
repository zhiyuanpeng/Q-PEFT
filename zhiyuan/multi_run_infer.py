#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   multi_run_infer.py
@Time    :   2024/02/18 03:21:23
@Author  :   Zhiyuan Peng@Santa Clara University
@Version :   1.0
@Desc    :   None
'''

import subprocess
import argparse
import os
from os.path import join, dirname
cwd = os.getcwd()
zhiyuan_dir = join(cwd, "xuyang")
zhiyuan_exp = join(zhiyuan_dir, "experiments")

def multirun(args):
    arg_dict = {}
    for key, vals in args._get_kwargs():
        arg_dict[key] = vals
    for checkpoint_path in arg_dict["checkpoint_name"]:
        for retriever in arg_dict["retriever_name"]:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')[:7]
            out_path = join(dirname(checkpoint_path), f"{commit}_test_clean_b{args.batch_size}")
            state = join(out_path, "{}_top_{}_sample_{}_fp16_{}_state.json".format(retriever, args.retriever_topk, args.sample_num, args.fp16))
            if not os.path.exists(state):
                if args.fp16:
                    train_command = ["python", "zhiyuan/infer.py", "--id", str(args.id), "--batch_size", str(arg_dict["batch_size"]), "--retriever_topk", str(args.retriever_topk), "--sample_num", str(args.sample_num), "--checkpoint_name", checkpoint_path, "--retriever_name", retriever, "--fp16"]
                else:
                    train_command = ["python", "zhiyuan/infer.py", "--id", str(args.id), "--batch_size", str(arg_dict["batch_size"]), "--retriever_topk", str(args.retriever_topk), "--sample_num", str(args.sample_num), "--checkpoint_name", checkpoint_path, "--retriever_name", retriever]
                print(" ".join(train_command))
                subprocess.call(train_command)
            else:
                print(f"{state} already done")
  

def main():
    parser = argparse.ArgumentParser(description='Training Starts ...')
    parser.add_argument("--id", type=int, default=0, help="device id")
    parser.add_argument("--batch_size", type=int, default=6, help="topk items")
    parser.add_argument("--retriever_topk", type=int, default=100, help="topk items")
    parser.add_argument("--sample_num", type=int, default=300, help="test sample num")
    parser.add_argument('-cn','--checkpoint_name', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-rn','--retriever_name', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument("--fp16", action="store_true")
    
    args = parser.parse_args()
    multirun(args)

if __name__ =="__main__":
    main()