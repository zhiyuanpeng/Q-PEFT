import os
from os.path import join, exists, dirname, basename
import json
from pprint import pprint
import jsonlines
from tqdm import tqdm
# cwd = dirname(os.getcwd())
cwd = os.getcwd()
dpr_root = join(cwd, 'data', 'DPR', 'downloads', 'data', 'retriever')
upr_root = join(cwd, 'data', 'UPR', 'downloads', 'data', 'retriever-outputs')

def show_example(file_path):
    file_name = basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f'{file_name} has {len(data)} examples. Show first one:\n')
    pprint(list(data[0].keys()))
    if file_name.split('-')[-1] != 'test':
        for d in data:
            d['question']
            d['positive_ctxs']
            d['negative_ctxs']
            d['hard_negative_ctxs']

def main():
    # check data
    for dataset_name in tqdm(['nq', 'squad1', 'webq', 'trivia']):
        train_path = join(dpr_root, f'{dataset_name}-train.json')
        show_example(train_path)
        dev_path = join(dpr_root, f'{dataset_name}-dev.json')
        show_example(dev_path)
        test_bm25_path = join(upr_root, 'bm25', f'{dataset_name}-test.json')
        show_example(test_bm25_path)
        test_contriever_path = join(upr_root, 'contriever', f'{dataset_name}-test.json')
        show_example(test_contriever_path)
        test_dpr_path = join(upr_root, 'dpr', f'{dataset_name}-test.json')
        show_example(test_dpr_path)
        test_mss_path = join(upr_root, 'mss', f'{dataset_name}-test.json')
        show_example(test_mss_path)
        test_mssdpr_path = join(upr_root, 'mss-dpr', f'{dataset_name}-test.json')
        show_example(test_mssdpr_path)

if __name__ == "__main__":
    main()

