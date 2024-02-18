from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
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
from zhiyuan.datamodules.components.dataset import JsonQADataset
from zhiyuan.datamodules.components.dpr_wiki_dataset import load_wiki
from zhiyuan.datamodules.components.openqa_dataset import OpenQADataset
from zhiyuan.utils.data_util import pointwise_collate_func, pairwise_collate_func, listwise_collate_func, infer_collate_func
from zhiyuan.utils.vendor_util import seed_everything

class DataModule():

    def __init__(
        self,
        batch_size: int = 32,
        q_num_per_batch: int = 32,
        num_workers: int = 0,
        max_seq_len: int = 300,
        dataset_name: str = "nq",
        wiki_path: str = None,
        pos_num: int = 4,
        hard_neg_num: int = 1,
        random_neg_num: int = 0,
        tokenizer=None,
        fixed_prompt: str = "",
        training_method: str = "pointwise",# pointwise, pairwise, listwise
        shuffle_neg: bool = True,
        disable_in_batch_neg: bool = True,
        doc_specific: bool = False,
        test_clean: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.batch_size=batch_size
        self.q_num_per_batch=q_num_per_batch
        self.num_workers=num_workers
        self.max_seq_len=max_seq_len
        self.dataset_name=dataset_name
        self.wiki_path=wiki_path
        self.wiki_id2text=None
        self.pos_num=pos_num
        self.hard_neg_num=hard_neg_num
        self.random_neg_num=random_neg_num
        self.tokenizer=tokenizer
        self.fixed_prompt=fixed_prompt
        self.training_method=training_method
        self.shuffle_neg=shuffle_neg
        self.disable_in_batch_neg=disable_in_batch_neg
        self.doc_specific=doc_specific
        self.test_clean=test_clean
        
    def train_dev_collate_func(self, data):
        if self.training_method == "pointwise":
            return pointwise_collate_func(self, data)
        elif "pairwise" in self.training_method:
            return pairwise_collate_func(self, data)
        elif self.training_method == "listwise":
            return listwise_collate_func(self, data)
        else:
            raise ValueError("Unrecognised training method")
    
    def test_collate_func(self, data):
        return infer_collate_func(self, data)

    def train_dataloader(self, sample_num: int=-1):
        g = torch.Generator()
        g.manual_seed(42)
        self.train_file = join(cwd, 'data', 'SPRK', self.dataset_name, f'{self.dataset_name}-train.json')
        self.data_train = JsonQADataset(self.train_file)
        self.data_train.load_data(sample_num)
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.q_num_per_batch,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=self.train_dev_collate_func,
            generator=g,
        )

    def dev_dataloader(self, sample_num: int=-1):
        seed_everything(42)
        self.dev_file = join(cwd, 'data', 'SPRK', self.dataset_name, f'{self.dataset_name}-dev.json')
        self.data_dev = JsonQADataset(self.dev_file)
        self.data_dev.load_data(sample_num)
        return DataLoader(
            dataset=self.data_dev,
            batch_size=self.q_num_per_batch,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.train_dev_collate_func,
        )
    
    def test_dataloader(self, retriever_name:str, sample_num:int=300, retriever_topk:int=1000):
        # lazy load
        seed_everything(42)
        if not self.wiki_id2text:
            self.wiki_id2text = load_wiki(self.wiki_path)
        self.test_file = join(cwd, 'data', 'SPRK', 'retriever-outputs', retriever_name, f'{self.dataset_name}-test.json')
        self.data_test = OpenQADataset(self.wiki_id2text, self.test_file, sample_num, retriever_topk)
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.test_collate_func,
        ), self.data_test.samples_q
    
def main():
    # train, dev test
    # data_module = DataModule()
    # train_loader = data_module.train_dataloader()
    # for batch in train_loader:
    #     pass
    # test dataset test
    from tqdm import tqdm
    wiki_path = join(cwd, 'data', 'SPRK', 'wikipedia-split', 'psgs_w100.tsv')
    retriever_topk = 100
    for dataset_name in ['nq', 'squad1', 'trivia', 'webq']:
        data_module = DataModule(batch_size=4, q_num_per_batch=32, max_seq_len=300, dataset_name=dataset_name, wiki_path=wiki_path, hard_neg_num=1, random_neg_num=0, tokenizer=None, )
        for retriever_name in ['bm25', 'contriever', 'dpr', 'mss', 'mss-dpr']:
            test_loader = data_module.test_dataloader(retriever_name=retriever_name, sample_num=1.0, retriever_topk=retriever_topk)
                

if __name__ == "__main__":
    main()

