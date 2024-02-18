#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   train_infer.py
@Time    :   2024/02/18 03:21:30
@Author  :   Zhiyuan Peng@Santa Clara University
@Version :   1.0
@Desc    :   None
'''

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
import time
from os.path import join
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
if join(cwd, 'xuyang') not in sys.path:
    sys.path.insert(0, join(cwd, 'xuyang'))
if join(cwd, 'zhiyuan') not in sys.path:
    sys.path.insert(0, join(cwd, 'zhiyuan'))
print(sys.path)

from xuyang.loss import sprk_compute_loss, pairwise_compute_loss
from zhiyuan.datamodules.datamodule import DataModule
from zhiyuan.utils.vendor_util import seed_everything, reset_args, AverageMeter, setup_train, setup_logger
import shutil
import subprocess

seed_everything(42)

def test_load():
    from peft import PeftModel, PeftConfig
    peft_checkpoint_dir = join(cwd, "zhiyuan", "checkpoints")
    checkpoint_name="nq/pairwise_v2_train_320_eval_128_fp32/qperbatch_4_batchsize_4/spt_True_l0/docspec_True_l2/exp_2023-12-13_2/epoch_1"
    peft_checkpoint_path = join(peft_checkpoint_dir, checkpoint_name)
    model_name_or_path = join(cwd, "llm/hf/llama/v2/llama-2-7b-chat")
    tokenizer_name_or_path = join(cwd, "llm/hf/llama/v2/llama-2-7b-chat")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    config = PeftConfig.from_pretrained(peft_checkpoint_path)
    config.base_model_name_or_path = model_name_or_path
    config.tokenizer_name_or_path = tokenizer_name_or_path
    model = PeftModel.from_pretrained(model, model_id=peft_checkpoint_path, config=config, fp16=True)

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def compute_loss(outputs, labels, num_virtual_tokens, train_method):
    if train_method == 'pointwise':
        return outputs.loss
    elif train_method == 'pairwise':
        return pairwise_compute_loss(outputs, labels, num_virtual_tokens)
    elif train_method == 'pairwise_v2':
        return sprk_compute_loss(outputs, labels, num_virtual_tokens)

def data2device(mini_batch, device):
    """convert a batch data to device

    Args:
        mini_batch (_type_): _description_
        device (_type_): _description_
    """
    mini_batch_device = {}
    for k, v in mini_batch.items():
        if v is not None:
            if type(v) == dict:
                docspec = {}
                for inner_k, inner_v in v.items():
                    docspec[inner_k] = inner_v.to(device)
                mini_batch_device[k] = docspec
            else:
                mini_batch_device[k] = v.to(device)
    return mini_batch_device

def main(args):
    train_flag = True
    if not args.spt:
        args.spt_layer_num = 0
    if not args.docspec:
        args.docspec_layer_num = 0
    if args.fix_docspec_embed:
        args.docspec_layer_num = 0
    tokenizer = load_tokenizer(args)
    args.num_virtual_tokens = len(tokenizer(args.prompt_tuning_init_text)["input_ids"])-1
    export_root, args = setup_train(args)
    log_writer = SummaryWriter(export_root)
    # logging info
    logger = setup_logger('train_logger', join(export_root, 'training_log.txt'))
    logger.info("start logging info for training")
    # creating model
    peft_config = PromptTuningConfig(
        spt_layer_num=args.spt_layer_num,
        docspec_layer_num=args.docspec_layer_num,
        docspec=args.docspec,
        spt=args.spt,
        fp16train=args.fp16train,
        fix_docspec_embed=args.fix_docspec_embed,
        exp_k=args.exp_k,
        exp_mode=args.exp_mode,
        exp_head_num=args.exp_head_num,
        exp_loc=args.exp_loc,
        exp_normalize=args.exp_normalize,
        exp_random=args.exp_random,
        exp_filter_dict_path=args.exp_filter_dict_path,
        exp_filter_mode=args.exp_filter_mode,
        exp_prompt=args.exp_prompt,
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=args.num_virtual_tokens,
        prompt_tuning_init_text=args.prompt_tuning_init_text,
        tokenizer_name_or_path=args.model_name_or_path,
    )
    if args.fp16train:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = get_peft_model(model, peft_config)
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")
    model = model.to(args.device)
    if train_flag and not args.spt:
        if not args.docspec:
            train_flag = False
        elif args.fix_docspec_embed:
            train_flag = False 
    if not train_flag:
        logger.info("no need to train")
        pre_best_model_path = Path(export_root).joinpath("epoch_{}".format(0))
        model.save_pretrained(pre_best_model_path)
        return str(pre_best_model_path)
    # dataset
    data_module = DataModule(
        tokenizer=tokenizer, 
        training_method=args.training_method, 
        batch_size=args.batch_size, 
        q_num_per_batch=args.q_num_per_batch, 
        max_seq_len=args.max_input_length, 
        fixed_prompt=args.fixed_prompt,
        dataset_name=args.dataset_name, 
        doc_specific=args.docspec,
        pos_num=args.pos_num,
        hard_neg_num=args.hard_neg_num,
        random_neg_num=args.random_neg_num,
        disable_in_batch_neg=args.disable_in_batch_neg)
    train_dataloader = data_module.train_dataloader(sample_num=args.training_sample)
    eval_dataloader = data_module.dev_dataloader(sample_num=args.eval_sample_num)

    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.enable_scheduler:
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_dataloader) * args.num_epochs),
        )
    # training and evaluation
    original_eval_loss = 999999
    early_stop_epoch = 0
    pre_best_model_path = None
    current_step = 0
    for epoch in range(args.num_epochs):
        if not train_flag:
            break
        if early_stop_epoch > 5:
            logger.info('Terminating because of early stopping!')
            break
        avg_train_loss = AverageMeter()
        avg_val_loss = AverageMeter()
        model.train()
        # 
        for step, batch in enumerate(tqdm(train_dataloader)):
            # batch = {k: v.to(args.device) for k, v in batch.items()}
            # outputs = model(**batch)
            for mini_batch in batch:
                mini_batch = data2device(mini_batch, args.device)
                outputs, input_labels, d_k_ids = model(**mini_batch)
                loss = compute_loss(outputs, input_labels, args.num_virtual_tokens, args.training_method)
                log_writer.add_scalar('Training/train_loss_step', loss.detach().float().item(), current_step)
                current_step += 1
                avg_train_loss.update(loss.detach().float().item())
                loss.backward()
                optimizer.step()
                if args.enable_scheduler:
                    lr_scheduler.step()
                optimizer.zero_grad()
        # evaluate eval dataset
        for step, batch in enumerate(tqdm(eval_dataloader)):
            for mini_batch in batch:
                mini_batch = data2device(mini_batch, args.device)
                with torch.no_grad():
                    outputs, input_labels, d_k_ids = model(**mini_batch)
                loss = compute_loss(outputs, input_labels, args.num_virtual_tokens, args.training_method)

                avg_val_loss.update(loss.detach().float().item())
        # get metrics
        # test model mode for saving all the models
        filepath = Path(export_root).joinpath("epoch_{}".format(epoch))
        # model.save_pretrained(filepath)
        # saving model
        if avg_val_loss.avg < original_eval_loss:
            original_eval_loss = avg_val_loss.avg
            early_stop_epoch = 0
            logger.info('new best val loss, model saved')
            model.save_pretrained(filepath)
            if pre_best_model_path is not None:
                shutil.rmtree(pre_best_model_path)
            pre_best_model_path = filepath
        else:
            early_stop_epoch += 1
        # logger
        log_writer.add_scalar('Training/train_loss_epoch', avg_train_loss.avg, epoch)
        log_writer.add_scalar('Training/val_loss_epoch', avg_val_loss.avg, epoch)
        logger.info(f"{epoch=}: {avg_train_loss.avg=} {avg_val_loss.avg=}")
    logger.info("Training Done!")
    log_writer.close()
    
    # Free up GPU memory
    return str(pre_best_model_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # PEFT related, less likely to change, mostly use default values
    parser.add_argument("--llm_name", type=str, default="llama-v2-7b-chat", help="llm's name")

    # Training related, less likely to change, mostly use default values
    parser.add_argument("--fp16train", action="store_true", help="true to train fp16 model")
    parser.add_argument("--max_input_length", type=int, default=300, help="max input length")
    parser.add_argument("--lr", type=float, default=3e-2, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=20, help="# of training epochs")
    parser.add_argument("--eval_sample_num", type=int, default=128, help="# of quereis sampled from eval data for eval after each training epoch")
    parser.add_argument("--fixed_prompt", type=str, default="", help="example prompt to instruct the llm")


    """Training related, likely to chagne for experiments
    for each batch, we first sample q_num_per_batch queries, then according to loss function, we collect the pos and neg documents. So, for a batch of q_num_per_batch queries, there are likely more than q_num_per_batch document-query pairs. To avoid GPU out of memory error, for these document-query pairs, we further split them into mini batches each of which has batch_size pairs
    """
    parser.add_argument("--training_sample", type=int, default=320, help="# of quereis sampled from train for training")
    parser.add_argument("--batch_size", type=int, default=4, help="# of queries consumed by GPU for each mini batch")
    parser.add_argument("--q_num_per_batch", type=int, default=4, help="# of queries in each batch before splitting into mini batches")
    parser.add_argument("--ml_name", type=str, default="spt", help="experiment name")
    
    # frequently change to do experiments
    parser.add_argument("--dataset_name", type=str, default="nq", help="dataset name")
    parser.add_argument("--training_method", type=str, default="pairwise_v2", choices=["pointwise", "pairwise", "pairwise_v2"], help="training method name")
    parser.add_argument("--docspec", action="store_true", help="doc specific or not")
    parser.add_argument("--docspec_layer_num", type=int, default=0, help="layer num of doc spec")
    parser.add_argument("--fix_docspec_embed", action="store_true", help="do not update docspec embed layer, must set docspec_layer_num 0")
    parser.add_argument("--spt", action="store_true", help="spt or not")
    parser.add_argument("--spt_layer_num", type=int, default=0, help="layer num of doc spec")
    parser.add_argument("--num_virtual_tokens", type=int, default=50, help="num virtual tokens for prompt")
    parser.add_argument("--prompt_tuning_init_text", type=str, default="given the hints, please generate a question for the input passage", help="initialized hard prompt")
    # dataset related
    parser.add_argument("--pos_num", type=int, default=1, help="DPR only use 1 positive doc") 
    parser.add_argument("--hard_neg_num", type=int, default=1, help="DPR only use 1 hard negative doc+inbatch negative") 
    parser.add_argument("--random_neg_num", type=int, default=0, help="# of normal negative docs") 
    parser.add_argument("--disable_in_batch_neg", action="store_true", help="in batch neg or not, disabled for pointwise")
    # for GPU
    parser.add_argument("--device_idx", type=str, default="0", help="device id")
    #
    parser.add_argument("--debug", action="store_true", help="score checkpoints in debug folder otherwise in checkpoints folder")
    parser.add_argument("--enable_scheduler", action="store_true", help="set to enable scheduler")
    # query expansion
    parser.add_argument("--exp_k", type=int, default=1, help="# of expanded tokens") 
    parser.add_argument("--exp_mode", type=int, default=1, help="set 1: for each token, retrieve exp_k tokens; set 2: for each query expand to top exp_k tokens")
    parser.add_argument("--exp_head_num", type=int, default=0, help="for mode 4, attention haed num")
    parser.add_argument("--exp_loc", type=str, default="prefix", help="append expanded tokens to prefix or suffix the document", choices=["prefix", "suffix"])
    parser.add_argument("--exp_normalize", action="store_true", help="Set True to apply normalize on q_v matrix")
    parser.add_argument("--exp_random", action="store_true", help="Set True to select random k tokens from V for mode 2 and from document for mode 3")
    parser.add_argument("--exp_filter_dict_path", default="./token_filter.pkl",help="blacklist of tokens to be filtered out")
    parser.add_argument("--exp_filter_mode", default="d", help="n: None, q: mode3, remove q itself, d: duplicate, ds: duplicate and stopword, da: duplicate and alpha, dsa: duplicate, stopword and alpha")
    parser.add_argument("--exp_prompt", type=str, default="hints", help="prompt before the retrieved topk tokens")
    # inference related
    parser.add_argument("--retriever_topk", type=int, default=100, help="topk items")
    parser.add_argument('-rn','--retriever_name', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument("--test_sample_num", type=int, default=300, help="# of teset queries")
    args = parser.parse_args()
    # use your own reset_args funciton to reset llm path in case llm models are stored in different locations in different 
    args = reset_args(args)
    pre_best_model_path = main(args)
    pre_best_model_path = pre_best_model_path.replace(cwd, "", 1)[1:]
    torch.cuda.empty_cache()
    time.sleep(10)
    # call infer
    for retriever in args.retriever_name:
        infer_command = ["python", "zhiyuan/infer.py", "--id", str(args.device_idx), "--batch_size", "6", "--retriever_topk", str(args.retriever_topk), "--sample_num", str(args.test_sample_num), "--checkpoint_name", str(pre_best_model_path), "--retriever_name", retriever, "--fp16"]
        print(" ".join(infer_command))
        subprocess.call(infer_command)

    
    