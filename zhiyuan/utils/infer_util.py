#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   infer_util.py
@Time    :   2023/12/19 14:05:10
@Author  :   Zhiyuan Peng@Santa Clara University
@Version :   1.0
@Desc    :   helper functions for inference
'''

from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import json
from os.path import dirname, join
from typing import List
from scipy import stats

def calculate_topk_hits(scores: List[List[bool]], max_k: int):
        top_k_hits = [0] * max_k
        for question_hits in scores:
            best_hit = next((i for i, x in enumerate(question_hits[:max_k]) if x), None)
            if best_hit is not None:
                top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
        return top_k_hits
    
def compute_topk_hit_rate(answers_list: List[List[bool]], report_top_k=[20, 100], string_prefix="Original Ranking"):
    topk_hits = calculate_topk_hits(answers_list, max_k=report_top_k[-1])
    n_docs = len(answers_list)
    topk_hits = [i/n_docs for i in topk_hits]
    print(string_prefix)
    for i in report_top_k:
        print("HR@{}: {:.2f}".format(i, topk_hits[i - 1] * 100))
    print("\n")

def sig_test_hit_rate(logger, retriever_list: List[List[bool]], re_ranker_list: List[List[bool]], report_top_k: List[int]):
    
    sig_test = {}
    for k in report_top_k:
        # Convert HR@k into binary outcomes for all queries
        retriever_hits = [1 if any(q_hits[:k]) else 0 for q_hits in retriever_list]
        re_ranker_hits = [1 if any(q_hits[:k]) else 0 for q_hits in re_ranker_list]
        
        # Perform a significance test, such as a Chi-squared test for binary data
        try:
            chi2_stat, p_value, _, _ = stats.chi2_contingency(np.array([
                [retriever_hits.count(1), retriever_hits.count(0)],
                [re_ranker_hits.count(1), re_ranker_hits.count(0)]
            ]))
            logger.info(f"HR@{k} Chi-squared: chi2_stat={chi2_stat}, p-value={p_value}")
        except:
            logger.info(f"can't compute Chi-squared test due to zeros in inputs")
            
        rel_t_stat, rel_p_value = stats.ttest_rel(re_ranker_hits, retriever_hits)
        # ind_t_stat, ind_p_value = stats.ttest_ind(re_ranker_hits, retriever_hits)
        # sig_test[k] = {"rel": [rel_t_stat, rel_p_value], "ind": [ind_t_stat, ind_p_value]}
        sig_test[k] = {"rel": [rel_t_stat, rel_p_value]}
    logger.info("\n")
    for k in report_top_k:    
        logger.info(f"HR@{k} ttest_rel: t-stat={sig_test[k]['rel'][0]}, p-value={sig_test[k]['rel'][1]}")
    logger.info("\n")
    
    # for k in report_top_k:
    #     logger.info(f"HR@{k} ttest_ind: t-stat={sig_test[k]['ind'][0]}, p-value={sig_test[k]['ind'][1]}")
    logger.info("\n")

def compute_topk_hit_rate_logger(logger, answers_list, report_top_k=[20, 100], string_prefix="Original Ranking"):
    topk_hits = calculate_topk_hits(answers_list, max_k=report_top_k[-1])
    n_docs = len(answers_list)
    topk_hits = [i/n_docs for i in topk_hits]
    if logger:
        logger.info(string_prefix)
        for i in report_top_k:
            logger.info("HR@{}: {:.2f}".format(i, topk_hits[i - 1] * 100))
        logger.info("\n")
    else:
        print(string_prefix)
        for i in report_top_k:
            print("HR@{}: {:.2f}".format(i, topk_hits[i - 1] * 100))
        print("\n")

def calculate_recall_at_k(true_labels: List[bool], sorted_labels: List[bool], k: int) -> float:
    # Get the top k sorted labels
    top_k_sorted_labels = sorted_labels[:k]
    # Count the number of true positives in the top k sorted labels
    true_positives = sum(top_k_sorted_labels)
    # Count the total number of relevant documents
    total_relevant = sum(true_labels)
    # Calculate recall at k
    return true_positives / total_relevant if total_relevant else 0.0

def evaluate_recall(logger, true_list: List[List[bool]], pred_list: List[List[bool]], report_top_k: List[int], retriever_name="bm25_scu") -> None:
    recalls_re_ranker = {k: [] for k in report_top_k}
    recalls_retriever = {k: [] for k in report_top_k}

    # Calculate recalls for re-ranker
    for true_labels, sorted_labels in zip(true_list, pred_list):
        for k in report_top_k:
            recall = calculate_recall_at_k(true_labels, sorted_labels, k)
            recalls_re_ranker[k].append(recall)
    
    # Calculate recalls for retriever (assuming the retriever's results are the original order)
    for true_labels in true_list:
        for k in report_top_k:
            recall = calculate_recall_at_k(true_labels, true_labels, k)
            recalls_retriever[k].append(recall)
    ans = {}
    # Perform significance tests
    logger.info(f"Pred recall: {retriever_name}")
    for k in report_top_k:
        logger.info("Recall@{}: {:.2f}".format(k, sum(recalls_re_ranker[k])*100/len(recalls_re_ranker[k])))
        ans[f"R_{k}"] = sum(recalls_re_ranker[k])*100/len(recalls_re_ranker[k])
    logger.info("\n")
    top2k = [5, 10]
    top3k = [5, 10, 20]
    ans["R_top2"] = sum([ans[f"R_{k}"] for k in top2k])/2
    logger.info("Recall@top2: {:.2f}".format(ans["R_top2"]))
    ans["R_top3"] = sum([ans[f"R_{k}"] for k in top3k])/3
    logger.info("Recall@top3: {:.2f}".format(ans["R_top3"]))
    ans["R_all"] = sum([ans[f"R_{k}"] for k in report_top_k])/len(report_top_k)
    logger.info("Recall@all: {:.2f}".format(ans["R_all"]))
    logger.info(f"Original recall: {retriever_name}")
    for k in report_top_k:
        logger.info("Recall@{}: {:.2f}".format(k, sum(recalls_retriever[k])*100/len(recalls_retriever[k])))
    logger.info("\n")
    
    for k in report_top_k:
        rel_t_stat, rel_p_value = stats.ttest_rel(recalls_re_ranker[k], recalls_retriever[k])
        logger.info(f"Recall@{k} ttest_rel: t-stat={rel_t_stat}, p-value={rel_p_value}")
        ans[f"P_{k}"] = rel_p_value
    logger.info("\n")
    return ans

    # for k in report_top_k:
    #     ind_t_stat, ind_p_value = stats.ttest_ind(recalls_re_ranker[k], recalls_retriever[k])
    #     logger.info(f"Recall@{k} ttest_ind: t-stat={ind_t_stat}, p-value={ind_p_value}")
    # logger.info("\n")

def extract_pred_labels(pred_score, gt_score):
    """copy from xuyang's compute_hits

    Args:
        pred_score (_type_): _description_
        gt_score (_type_): _description_

    Returns:
        _type_: _description_
    """
    reranking_hits = []
    pred_reranking_index = np.argsort(np.array(pred_score))
    for i in range(len(pred_score)):
        if gt_score[pred_reranking_index[i]] is True:
            reranking_hits.append(True)
        else:
            reranking_hits.append(False)
    return reranking_hits

def compute_loss(outputs, labels, virtual_token_num, margin=0):
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = CrossEntropyLoss(reduction='none')
    shift_labels = shift_labels.to(shift_logits.device)
    # loss.shape = (1,4)
    loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
    loss_mask = loss != 0
    loss = (loss*loss_mask).sum(dim=1)/loss_mask.sum(dim=1)
    return loss

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

def get_expanded_query(d_k_ids, vocab_dict):
    """get expaned query from d_k_ids

    Args:
        d_k_ids (_type_): _description_
        tokenizer (_type_): _description_

    Returns:
        _type_: _description_
    """
    expanded_query = []
    for d_k_id in d_k_ids:
        for id in d_k_id:
            expanded_query.append(vocab_dict[id.item()])
        break
    return expanded_query

def inference(model, data_module, dataset_name, device, json_file_path, logger, sample_num, retriever_topk, virtual_token_num, retriever_name, fp16, doc_specific=False, report_top_k=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]): 
    vocab_dict = data_module.tokenizer.get_vocab()
    vocab_dict = {v: k for k, v in vocab_dict.items()}
    test_loader, samples_q = data_module.test_dataloader(retriever_name=retriever_name, sample_num=sample_num, retriever_topk=retriever_topk)
    # write sampled queries to file
    q_dir_name = dirname(json_file_path)
    q_out_path = join(q_dir_name, f"{retriever_name}_top_{retriever_topk}_sample_{sample_num}_fp16_{fp16}_queries.txt")
    cleaned_q_out_path = join(q_dir_name, f"{retriever_name}_top_{retriever_topk}_sample_{sample_num}_fp16_{fp16}_queries_cleaned.txt")
    with open(q_out_path, "w") as q_outfile:
        for s_q in samples_q:
            q_outfile.write(s_q+"\n")
    # save pred results and save to json
    start = 0
    pred_json_results = defaultdict(list)
    pred_json_results_path = join(q_dir_name, f"{retriever_name}_top_{retriever_topk}_sample_{sample_num}_fp16_{fp16}_pred_results.json")
    #
    pred_answers_list = []
    retriever_answers_list = []
    cleaned_samples_q = []
    for (batch_inputs, batch_answers, batch_items, batch_data) in tqdm(test_loader):
        """ read query by batch
        batch = inputs, answers
        inputs:  batch_size queries each of which has retriever_topk items
        answers: the corresponding true label of inputs
        """
        for single_inputs, single_answers, single_items, single_data in zip(batch_inputs, batch_answers, batch_items, batch_data):
            """each query has retriever_topk items splited into batches

            Args:
                single_inputs (_type_): retriever_topk items belongs to one query
                single_answers (_type_): labels of single_inputs
            """
            ans = []
            ans_gt = []
            pred_json_results[retriever_name].append({"question": single_data['question']})
            cleaned_samples_q.append(single_data['question'])
            for inputs, answers, ori_items in zip(single_inputs, single_answers, single_items):
                """retriever_topk items are splited into batches, batch_size is len(batch_inputs)

                Args:
                    inputs (_type_): len(batch_inputs) tokenized doc+query
                    answers (_type_): labels
                    ori_items (_type_): batched items, for example retriever_topk=100, batch_size=4, then len(ori_items)=4 and we need to execute this for loop 25 times
                """
                # print(inputs)
                with torch.no_grad():
                    # inputs = {k: v.to(device) for k, v in inputs.items()}
                    # outputs = model(**inputs)
                    inputs = data2device(inputs, device)
                    outputs, input_labels, d_k_ids = model(**inputs) #docspec_embeddings.weight.
                    loss = compute_loss(outputs, input_labels, virtual_token_num)
                    pred_scores = list(loss.cpu().numpy())
                    # loss = outputs.loss.item()
                    # print(loss)
                    # pred_scores = [loss]
                    ans.extend(pred_scores)
                    ans_gt.extend(answers)
                    # for item, pred_score in zip(ori_items, pred_scores):
                    #     item['sprk_score'] = str(pred_score)
                    #     pred_json_results[retriever_name][start]['items'].append(item)
                    assert len(ans) == len(ans_gt)
            assert len(ans) == retriever_topk
            pred_ans = extract_pred_labels(ans, ans_gt)
            pred_answers_list.append(pred_ans)
            # retriever answers
            retriever_answers_list.append(ans_gt)
            # extract expanded query
            if doc_specific:
                pred_json_results[retriever_name][start]['expanded_query'] = get_expanded_query(d_k_ids, vocab_dict)
            # update ranked items
            pred_json_results[retriever_name][start]['items'] = pred_ans 
            start += 1
        # break
    assert len(pred_answers_list) == len(retriever_answers_list)
    # assert len(pred_answers_list) == sample_num
    print(f"{dataset_name} {retriever_name}: ")
    logger.info("clean ranking results: ")

    ans = evaluate_recall(logger, retriever_answers_list, pred_answers_list, report_top_k=report_top_k, retriever_name=retriever_name)
    
    pred_results = compute_topk_hit_rate_logger(logger, pred_answers_list, report_top_k=report_top_k, string_prefix=f"Pred Ranking: {retriever_name}")

    retriever_results = compute_topk_hit_rate_logger(logger, retriever_answers_list, report_top_k=report_top_k, string_prefix=f"Original Ranking: {retriever_name}")

    sig_test_hit_rate(logger, retriever_answers_list, pred_answers_list, report_top_k=report_top_k)

    logger.info("full ranking results: ")
    full_retriever_answers_list = []
    full_pred_answers_list = []
    for i in range(len(retriever_answers_list)):
        full_retriever_answers_list.append(retriever_answers_list[i])
        full_pred_answers_list.append(pred_answers_list[i])
    if len(retriever_answers_list) < data_module.data_test.k:
        # same data are filtered out
        for i in range(data_module.data_test.k-len(retriever_answers_list)):
            full_retriever_answers_list.append([False]*retriever_topk)
            full_pred_answers_list.append([False]*retriever_topk)

    f_ans = evaluate_recall(logger, full_retriever_answers_list, full_pred_answers_list, report_top_k=report_top_k, retriever_name=retriever_name)

    pred_results = compute_topk_hit_rate_logger(logger, full_pred_answers_list, report_top_k=report_top_k, string_prefix=f"Pred Ranking: {retriever_name}")

    retriever_results = compute_topk_hit_rate_logger(logger, full_retriever_answers_list, report_top_k=report_top_k, string_prefix=f"Original Ranking: {retriever_name}")

    sig_test_hit_rate(logger, full_retriever_answers_list, full_pred_answers_list, report_top_k=report_top_k)
    
    # saving the results file
    # json_object = json.dumps(pred_json_results)
    with open(pred_json_results_path, "w") as outfile:
        json.dump(pred_json_results, outfile, indent=4)
    json_object = json.dumps({"state": "done"})
    dir_name = dirname(json_file_path)
    out_path = join(dir_name, f"{retriever_name}_top_{retriever_topk}_sample_{sample_num}_fp16_{fp16}_state.json")
    with open(out_path, "w") as outfile:
        outfile.write(json_object)
    # if data_module.test_clean:
    with open(cleaned_q_out_path, "w") as q_outfile:
        for s_q in cleaned_samples_q:
            q_outfile.write(s_q+"\n")
    return ans, f_ans

def main():

    from tqdm import tqdm
    import os
    from os.path import join
    import sys
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    if join(cwd, 'xuyang') not in sys.path:
        sys.path.insert(0, join(cwd, 'xuyang'))
    if join(cwd, 'zhiyuan') not in sys.path:
        sys.path.insert(0, join(cwd, 'zhiyuan'))
    from zhiyuan.datamodules.datamodule import DataModule
    wiki_path = join(cwd, 'data', 'SPRK', 'wikipedia-split', 'psgs_w100.tsv')
    retriever_topk = 100
    for dataset_name in ['nq', 'squad1', 'trivia', 'webq']:
        data_module = DataModule(batch_size=1, dataset_name=dataset_name, wiki_path=wiki_path)
        for retriever_name in ['bm25', 'contriever', 'dpr', 'mss', 'mss-dpr']:
            test_loader = data_module.test_dataloader(retriever_name=retriever_name, sample_rate=1.0, retriever_topk=retriever_topk)
            # acc, num = 0, 0
            # for batch in tqdm(test_loader):
            #     """
            #     sample = {'id': idx,
            #     'question': question,
            #     'answers': answers,
            #     'items': items}
            #     """
            #     num += 1
            #     items = batch[0]['items']
            #     for i in range(topk):
            #         if items[i]['has_answer']:
            #             acc += 1
            # recall = acc/(num*topk)
            # print(f'retriever {retriever_name} on {dataset_name} has recall: {recall*100}')
            answers_list = []
            for batch in tqdm(test_loader):
                """
                sample = {'id': idx,
                'question': question,
                'answers': answers,
                'items': items}
                """
                items = batch[0]['items']
                ans = [item['has_answer'] for item in items]
                answers_list.append(ans)
            compute_topk_hit_rate(answers_list, report_top_k=[10])