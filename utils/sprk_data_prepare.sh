# @File    :   sprk_download_data.py
# @Time    :   2023/08/16 16:09:45
# @Author  :   
# @Version :   1.0
# @Contact :   zpeng@scu.edu
# @License :   (C)Copyright 2020-2023 Zhiyuan Peng@Santa Clara University
# @Desc    :   generate sprk datasets

mkdir ./data/SPRK/
cd ./data/SPRK/
mkdir nq squad1 trivia webq

# for nq
cd nq 
cp ../../DPR/downloads/data/retriever/nq-train.json ./
cp ../../DPR/downloads/data/retriever/nq-dev.json ./

# for squad1
cd ..
cd squad1
cp ../../DPR/downloads/data/retriever/squad1-train.json ./
cp ../../DPR/downloads/data/retriever/squad1-dev.json ./

# for trivia
cd ..
cd trivia
cp ../../DPR/downloads/data/retriever/trivia-train.json ./
cp ../../DPR/downloads/data/retriever/trivia-dev.json ./

# for webq
cd ..
cd webq
cp ../../DPR/downloads/data/retriever/webq-train.json ./
cp ../../DPR/downloads/data/retriever/webq-dev.json ./

# for test
cd ..
cp -r ../UPR/downloads/data/* ./
