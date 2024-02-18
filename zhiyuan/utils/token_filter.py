#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   token_filter.py
@Time    :   2023/12/30 19:42:22
@Author  :   Zhiyuan Peng@Santa Clara University
@Version :   1.0
@Desc    :   filter the tokens that are not alphabetic in llama2's vocabulary
'''
import os
from os.path import join
from transformers import AutoTokenizer
import json
import re
import pickle

def is_english_letter(char):
    return bool(re.match(r'[A-Za-z]', char))


cwd = os.getcwd()
llma_path = join(cwd, "llm/hf/llama/v2/Llama-2-7b-chat-hf")

stopwords = ['a', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or', 's', 'such', 't', 'that', 'the', 'their', 'then', 'there', 'these', 'they', 'this', 'to', 'was', 'will', 'with', 'www']
stopwords = set(stopwords)

def find_non_alphabetic_keys(llma_path):
    tokenizer = AutoTokenizer.from_pretrained(llma_path)
    non_alphabetic_keys = set()
    stopword_keys = set()
    for key, value in tokenizer.vocab.items():
        if not any(is_english_letter(char) for char in key):
            non_alphabetic_keys.add(value)
        if key.lower() in stopwords:
            stopword_keys.add(value)
        if key.replace("_", "").lower() in stopwords:
            stopword_keys.add(value)
    token_filter = {"non_alphabetic_keys": non_alphabetic_keys, "stopword_keys": stopword_keys}
    with open(join(cwd, "token_filter.pkl"), "wb") as f:
        pickle.dump(token_filter, f)
    return non_alphabetic_keys, stopword_keys

def main():
    non_alphabetic_keys = find_non_alphabetic_keys(llma_path)
    print("done")

if __name__ == "__main__":
    main()

