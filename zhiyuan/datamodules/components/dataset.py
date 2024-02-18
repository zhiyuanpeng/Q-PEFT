import collections
import glob
import logging
import os
import sys
from os.path import join
import random
from typing import Dict, List, Tuple
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
if join(cwd, 'xuyang') not in sys.path:
    sys.path.insert(0, join(cwd, 'xuyang'))
if join(cwd, 'zhiyuan') not in sys.path:
    sys.path.insert(0, join(cwd, 'zhiyuan'))
from zhiyuan.utils.data_util import read_data_from_json_file, Dataset

logger = logging.getLogger(__name__)
BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])


def get_dpr_files(source_name) -> List[str]:
    if os.path.exists(source_name) or glob.glob(source_name):
        return glob.glob(source_name)
    # else, download files


class BiEncoderSample(object):
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]


class JsonQADataset(Dataset):
    """
    raw data format:
    [
        {
            "question": "....",
            "answers": ["...", "...", "..."],
            "positive_ctxs": [{
                "title": "...",
                "text": "...."
            }],
            "negative_ctxs": ["..."],
            "hard_negative_ctxs": ["..."]
        },
        ...
    ]
    after process, convert to 
    a list of BiEncoderSample objects, each of which has the following fields:
    query: str
    positive_passages: [{"text": str, "title": str}, ...]
    negative_passages: [{"text": str, "title": str}, ...]
    """
    def __init__(
        self,
        file: str,
        wiki: str = None,
        special_token: str = None,
        encoder_type: str = None,
        shuffle_positives: bool = False,
        normalize: bool = False,
        query_special_suffix: str = None,
        # tmp: for cc-net results only
        exclude_gold: bool = False,
    ):
        super().__init__(
            special_token=special_token,
            encoder_type=encoder_type,
            shuffle_positives=shuffle_positives,
            query_special_suffix=query_special_suffix,
        )
        self.file = file
        self.wiki = wiki
        self.normalize = normalize
        self.exclude_gold = exclude_gold

    def load_data(self, sample_num: int=-1):
        logger.info("Data file: %s", self.file)
        data = read_data_from_json_file(self.file)
        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total cleaned data size: %d", len(self.data))
        k = len(self.data) if sample_num == -1 else min(len(self.data), sample_num)
        self.data = random.sample(self.data, k)
        logger.info("Selecting subset # %d", k)

    def __getitem__(self, index) -> BiEncoderSample:
        json_sample = self.data[index]
        r = BiEncoderSample()
        r.query = self._process_query(json_sample["question"])

        positive_ctxs = json_sample["positive_ctxs"]
        if self.exclude_gold:
            ctxs = [ctx for ctx in positive_ctxs if "score" in ctx]
            if ctxs:
                positive_ctxs = ctxs

        negative_ctxs = json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        hard_negative_ctxs = json_sample["hard_negative_ctxs"] if "hard_negative_ctxs" in json_sample else []

        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        def create_passage(ctx: dict):
            return BiEncoderPassage(
                normalize_passage(ctx["text"]) if self.normalize else ctx["text"],
                ctx["title"],
            )

        r.positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
        r.negative_passages = [create_passage(ctx) for ctx in negative_ctxs]
        r.hard_negative_passages = [create_passage(ctx) for ctx in hard_negative_ctxs]
        return r

def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    if ctx_text.startswith('"'):
        ctx_text = ctx_text[1:]
    if ctx_text.endswith('"'):
        ctx_text = ctx_text[:-1]
    return ctx_text

def main():
    file = join(cwd, 'data', 'SPRK', 'nq', 'nq-train.json')
    dataset = JsonQADataset(file)
    dataset.load_data()
    print(file)
    pass

if __name__ == "__main__":
    main()

