import json
import random
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


class OpenQADataset(Dataset):
    """
    raw test data format:
    [
        {
            "question": "who sings does he love me with reba",
            "answers": ["Linda Davis"],
            "ctxs": [     
                        {
                            "id": 11828871,
                            "score": 18.3,
                            "has_answer": false
                        },
                        {
                            "id": 11828872,
                            "score": 14.7,
                            "has_answer": false,
                        },
                        {
                            "id": 11828866,
                            "score": 14.4,
                            "has_answer": true,
                        },
                        ...
                    ]
        },
    ...
    ]
    after process, convert to
    a list of samples, each of which has the following fields:
    id: int, range from 0 to len(all samples)
    question: str
    answers: str
    items: [{"id": id in wiki, "score": bm25 score relevant to query, "has_answer": true or false, "text": look up id in wiki}, ...]
    """
    def __init__(self, wiki_id2text, filepath:str, sample_num:int=-1, retriever_topk:int=1000):
        super().__init__()
        print_rank_0(f' > building reranking dataset for {filepath}:')
        self.wiki_id2text = wiki_id2text
        self.samples = self.load_dataset(filepath)
        self.samples = sorted(self.samples, key=lambda x: x["question"])
        print("Example test sample: \n")
        for x in self.samples[:5]:
            print(x["question"])
        for x in self.samples[-5:]:
            print(x["question"])
        self.retriever_topk = retriever_topk
        self.k = len(self.samples) if sample_num == -1 else min(len(self.samples), sample_num) 
        print(f"Sample {self.k} queires to test\n")
        self.samples = random.sample(self.samples, self.k)
        print("after sample: \n")
        for x in self.samples[:5]:
            print(x["question"])
        for x in self.samples[-5:]:
            print(x["question"])
        self.samples_q = [item["question"] for item in self.samples]
        print_rank_0('  >> total number of samples: {}'.format(len(self.samples)))

        if "trivia" in filepath or 'webq' in filepath or 'entity-questions' in filepath \
                or "BEIR" in filepath or "squad" in filepath:
            self.ques_punc = ""
        elif "nq" in filepath or "efficientqa" in filepath:
            self.ques_punc = "?"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        # These [CLS] and [SEP] tokens exist due to BERT tokenization, so we need to remove them
        if "[CLS]" and "[SEP]" in row['question']:
            row['question'] = " ".join(row['question'].split()[1:-1])
        question = row['question'] + self.ques_punc
        
        if 'ctxs' in row:
            items = row['ctxs'][:self.retriever_topk]
        elif 'contexts' in row:
            items = row['contexts'][:self.retriever_topk]
        for item in items:
            item['text'] = self.wiki_id2text[int(item['id'])]
        answers = row['answers']

        sample = {'id': idx,
                  'question': question,
                  'answers': answers,
                  'items': items}
        return sample

    @staticmethod
    def load_dataset(filepath):
        with open(filepath) as fp:
            data = json.load(fp)

        # condition for interfacing with pyserineni BM25 outputs
        if isinstance(data, dict):
            return list(data.values())
        else:
            return data

def main():
    
    pass

if __name__ == "__main__":
    main()


