<div align="center">

# [Q-PEFT]()

</div>

# What is it?
This is the implementation of `Q-PEFT: Query-dependent Parameter Efficient Fine-tuning for Text Reranking with Large Language Models`

# Install git-lfs
Please refer to this [link](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md) to install git-lfs before cloning this repo. Commands to install git-lfs on Linux systems are copied here:

```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

sudo apt-get install git-lfs
```
# Install Env

```python3.8.17
# 
python -m venv qpeft-v0.5.0
source ./venv/bin/python
pip install -r requirements.txt
cd ./vendor/peft-0.5.0
pip install -e .
# If error: Proxy URL had no scheme, should start with http:// or https://
unset http_proxy
unset https_proxy
```

# Prepare Datasets
```
# download DPR data
python utils/dpr_data_prepare.py --resource data --output_dir ./data/DPR
# download UPR data
python utils/upr_data_prepare.py --resource data --output_dir ./data/UPR
# generate SPRK data
bash utils/sprk_data_prepare.sh
# generate bm25_scu data to replace data/SPRK/retriever-outputs/bm25_scu
# first pull and run bm25 anserini docker
docker pull beir/pyserini-fastapi 
docker run -p 8000:8000 -it --name bm25 --rm beir/pyserini-fastapi:latest
# then run the following three command in order
python zhiyuan/bm25_scu.py --dataset_name trivia --overwritejsonl --reindex
python zhiyuan/bm25_scu.py --dataset_name squad1
python zhiyuan/bm25_scu.py --dataset_name nq
python zhiyuan/bm25_scu.py --dataset_name xx(custom dataset name)
```
<a id="Train&Dev-Data-Format"></a>
# Train&Dev Data Format
We utilize the Train and Dev data released by DPR, the following descriptopm is copied from [DPR](https://github.com/facebookresearch/DPR#retriever-input-data-format):

The default data format of the Retriever training data is JSON.
It contains pools of 2 types of negative passages per question, as well as positive passages and some additional information.

```
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
```

Elements' structure  for negative_ctxs & hard_negative_ctxs is exactly the same as for positive_ctxs.
The preprocessed data available for downloading also contains some extra attributes which may be useful for model modifications (like bm25 scores per passage). Still, they are not currently in use by DPR.

<a id="TopK-data-format"></a>
# TopK Data Format
We directly utilize the topK results by different retrievers released by UPR. The following data description is copied from [UPR](https://github.com/DevSinghSachan/unsupervised-passage-reranking#input-data-format):

#### Wikipedia Evidence Passages
We follow the [DPR convention](https://arxiv.org/abs/2004.04906) and segment the Wikipedia articles into 100-word long passages.
DPR's provided evidence file can be downloaded with the command
```python
python utils/download_data.py --resource data.wikipedia-split.psgs_w100
```
This evidence file contains tab-separated fields for passage id, passage text, and passage title. 

```bash
id  text    title
1   "Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusiv
ely from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained 
with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother's spokesman (""prophet"
") to the Pharaoh. Part of the Law (Torah) that Moses received from"    Aaron
2   "God at Sinai granted Aaron the priesthood for himself and his male descendants, and he became the first High Priest of the Israelites. Aaron died before the Israelites crossed
 the North Jordan river and he was buried on Mount Hor (Numbers 33:39; Deuteronomy 10:6 says he died and was buried at Moserah). Aaron is also mentioned in the New Testament of the Bib
le. According to the Book of Exodus, Aaron first functioned as Moses' assistant. Because Moses complained that he could not speak well, God appointed Aaron as Moses' ""prophet"" (Exodu
s 4:10-17; 7:1). At the command of Moses, he let"   Aaron
... ... ...
``` 

#### Top-K Retrieved Data
The input data format is JSON. Each dictionary in the json file contains one question, a list containing data of the top-K retrieved passages, and an (optional) list of possible answers.
For each top-K passage, we include the (evidence) id, has_answer , and (optional) retriever score attributes.
The `id` attribute is passage id from the Wikipedia evidence file, `has_answer` denotes if the passage text contains the answer span or not.
Following is the template of the .json file

```json
[
  {
    "question": "....",
    "answers": ["...", "...", "..."],
    "ctxs": [
              {
                "id": "....",
                "score": "...",
                "has_answer": "....",
              },
              ...
            ]
  },
  ...
]
```

An example when passages are retrieved using BM25 when queries using Natural Questions dev set.
```json
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
```

# Reproducibility
## NQ

## TriviaQA

## WebQ

## SQuAD



