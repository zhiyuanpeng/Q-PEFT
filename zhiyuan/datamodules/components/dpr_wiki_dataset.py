# coding=utf-8

"""Wikipedia dataset from DPR code for ORQA."""

import sys
import csv
from abc import ABC
import numpy as np
from torch.utils.data import Dataset
import os
import sys
from os.path import join
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
if join(cwd, 'xuyang') not in sys.path:
    sys.path.insert(0, join(cwd, 'xuyang'))
if join(cwd, 'zhiyuan') not in sys.path:
    sys.path.insert(0, join(cwd, 'zhiyuan'))
from zhiyuan.utils import print_rank_0
from tqdm import tqdm
# https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
    
def load_wiki(filename):
    print_rank_0(' > Processing {} ...'.format(filename))
    total = 0
    id2text = {}

    with open(filename) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)  # skip the headers
        for row in tqdm(reader):
            # file format: doc_id, doc_text, title
            doc_id = int(row[0])
            text = row[1]
            title = row[2]
            assert doc_id not in id2text
            id2text[doc_id] = text+title
            total += 1
            # if total % 100000 == 0:
            #     print_rank_0('  > processed {} rows so far ...'.format(total))
    print_rank_0(' >> processed {} samples.'.format(len(id2text)))
    return id2text
    
def main():
    wiki = join(cwd, 'data', 'SPRK', 'wikipedia-split', 'psgs_w100.tsv')
    test_wiki = load_wiki(wiki)
    pass

if __name__ == "__main__":
    main()
