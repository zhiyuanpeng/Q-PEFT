#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   bm25_scu.py
@Time    :   2023/09/20 13:00:07
@Author  :   
@Version :   1.0
@Contact :   zpeng@scu.edu
@License :   (C)Copyright 2020-2023 zpeng@scu.edu
@Desc    :   we find that dataset retriever-outpus/bm25 released by paper UPR have differnen queries as other four retrievers. We fix this issue by extracting the queries from other retrievers and run bm25 to retrieve top 1000 wiki passages
'''

"""
This example shows how to evaluate Anserini-BM25 in BEIR.
Since Anserini uses Java-11, we would advise you to use docker for running Pyserini. 
To be able to run the code below you must have docker locally installed in your machine.
To install docker on your local machine, please refer here: https://docs.docker.com/get-docker/

After docker installation, please follow the steps below to get docker container up and running:

1. docker pull beir/pyserini-fastapi 
2. docker run -p 8000:8000 -it --name msmarco --rm beir/pyserini-fastapi:latest
4. docker run -p 8002:8000 -it --name fiqa --rm beir/pyserini-fastapi:latest

Once the docker container is up and running in local, now run the code below.
This code doesn't require GPU to run.

"""
import pathlib, os, json
import logging
import requests
import random
import sys
from os.path import join
import argparse
import math
from tqdm import tqdm
import csv
####
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
if join(cwd, 'xuyang') not in sys.path:
    sys.path.insert(0, join(cwd, 'xuyang'))
if join(cwd, 'zhiyuan') not in sys.path:
    sys.path.insert(0, join(cwd, 'zhiyuan'))
data_dir = join(cwd, "data", "SPRK", "retriever-outputs")
ori_wiki_path = join(cwd, "data", "SPRK", "wikipedia-split", "psgs_w100.tsv")
file_name = join(data_dir, "bm25_scu", "nq-test.json")
# with open(file_name) as fp:
#         data = json.load(fp)
def load_wiki2json(input_filename, output_filename, overwritejsonl, num: float = float("inf")):
    total = 0
    id2text = {}
    if overwritejsonl:
        print(f"overwirte {output_filename}")
        fout = open(output_filename, 'w', encoding="utf-8")
    with open(input_filename) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)  # skip the headers
        for row in tqdm(reader):
            if total >= num:
                break
            # file format: doc_id, doc_text, title
            doc_id = int(row[0])
            text = row[1]
            title = row[2]
            assert doc_id not in id2text
            id2text[doc_id] = text+title
            data = {"id": doc_id, "title": title, "contents": text}
            if overwritejsonl:
                json.dump(data, fout)
                fout.write('\n')
            total += 1
    print(' >> processed {} samples.'.format(len(id2text)))
    if overwritejsonl:
        fout.close()
    return id2text

def extract_raw_test(file_name):
    """extract the raw test data from one of other four retrievers, here we use dpr:
    # [
    #   {
    #     "question": "who sings does he love me with reba",
    #     "answers": ["Linda Davis"],
    #     "ctxs": [     
    #               {
    #                "id": 11828871,
    #                "score": 18.3,
    #                "has_answer": false
    #               },
    #            ...
    #             ]
    #   },
    #   ...
    # ]
    then remove the ctxs, for later bm25 retrieve
    Args:
        file_name (_type_): _description_
    """
    # base_folder = dirname(file_name)
    with open(file_name) as fp:
        data = json.load(fp)
    test_queries, test_dataset = [], []
    for item in data:
        item["ctxs"] = []
        test_queries.append(item["question"])
        test_dataset.append(item)

    return test_dataset, test_queries, [i for i in range(len(test_dataset))]

def tag_label(wiki_id2text, topk_doc_ids, answers):
    ctxs = []
    for doc_id, doc_score in topk_doc_ids.items():
        ctx = {}
        ctx["id"] = int(doc_id)
        ctx["score"] = doc_score
        label = False
        for answer in answers:
            if answer.lower() in wiki_id2text[int(doc_id)].lower():
                label = True
        ctx["has_answer"] = label
        ctxs.append(ctx)
    return ctxs

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=False, default="msmarco", type=str)
parser.add_argument('--port', required=False, default=8000, type=int)
parser.add_argument('--overwritejsonl', action='store_true')
parser.add_argument('--reindex', action='store_true')
parser.add_argument('--topk', required=False, default=1000, type=int)
args = parser.parse_args()
#### Provide model save path
output_folder = join(data_dir, "bm25_scu")
input_folder = join(data_dir, "dpr")
os.makedirs(output_folder, exist_ok=True)
wiki_jsonl = join(output_folder, f"psgs_w100.jsonl")
#### Convert wiki corpus to Pyserini Format #####
wiki_id2text = load_wiki2json(ori_wiki_path, wiki_jsonl, args.overwritejsonl)
#### Download Docker Image beir/pyserini-fastapi ####
#### Locally run the docker Image + FastAPI ####
docker_wiki_pyserini = f"http://127.0.0.1:{args.port}"

if args.reindex:
    #### Upload Multipart-encoded files ####
    with open(wiki_jsonl, "rb") as fIn:
        r = requests.post(docker_wiki_pyserini + "/upload/", files={"file": fIn}, verify=False)

    #### Index documents to Pyserini #####
    index_name = f"beir/{args.dataset_name}" # beir/scifact
    r = requests.get(docker_wiki_pyserini + "/index/", params={"index_name": index_name})

#### Retrieve documents from Pyserini #####
test_dataset, query_texts, qids = extract_raw_test(join(input_folder, f"{args.dataset_name}-test.json"))
chunk_size = 5000
chunk_num = math.ceil(len(test_dataset)/chunk_size)

for i in tqdm(range(chunk_num)):
    if i != chunk_num - 1:
        payload = {"queries": query_texts[i*chunk_size: (i+1)*chunk_size], "qids": qids[i*chunk_size: (i+1)*chunk_size], "k": args.topk}
    else:
        payload = {"queries": query_texts[i*chunk_size:], "qids": qids[i*chunk_size:], "k": args.topk}
    #### Retrieve pyserini results (format of results is identical to qrels)
    results = json.loads(requests.post(docker_wiki_pyserini + "/lexical/batch_search/", json=payload).text)["results"]
    
    # iterate each q and retrieved topk doc list pair
    for q_id, topk_doc_ids in results.items():
        # sort and cut if tok < 1000, bm25 return 1000 items by default
        # topk_doc_ids = [(ss, ii) for ii, ss in topk_doc_ids.items()]
        # sorted(topk_doc_ids, reverse=True)
        # topk_doc_ids = topk_doc_ids[:args.topk]
        # topk_doc_ids = [ii for (_, ii) in topk_doc_ids]
        q_id = int(q_id)
        test_dataset[q_id]["ctxs"] = tag_label(wiki_id2text, topk_doc_ids, test_dataset[q_id]["answers"])
# write to 
with open(join(output_folder, f"{args.dataset_name}-test.json"), "a+") as fout:
    test_dataset_json = json.dumps(test_dataset)
    fout.write(test_dataset_json)