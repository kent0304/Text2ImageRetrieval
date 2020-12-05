import os
import re
import csv
import glob
import pathlib
import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm
import MeCab
import pickle
import random
from natsort import natsorted
from PIL import Image
from multiprocessing import Pool
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from model import TripletModel
from gensim.models import Word2Vec

# pickleで読み込み
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/train_dataset.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/valid_dataset.pkl', 'rb') as f:
    valid_dataset = pickle.load(f)
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)


def mydataset(dataset):
    # レシピコーパスで学習したWord2Vec
    text_model = Word2Vec.load("/mnt/LSTA5/data/tanaka/retrieval/text2image/word2vec.model")
    data_num = len(dataset)
    sentence_vec = []
    for data in tqdm(dataset, total=data_num):
        text = data[0]
        # 形態素解析
        mecab = MeCab.Tagger("-Owakati")
        token_list = mecab.parse(text).split()
        # 文全体をベクトル化
        sentence_sum = np.zeros(text_model.wv.vectors.shape[1], )
        for token in token_list:
            if token in text_model.wv:
                sentence_sum += text_model.wv[token]
            else:
                continue
        sentence = sentence_sum / len(token_list)
        sentence = torch.from_numpy(sentence).clone()
        sentence_vec.append(sentence)
    return sentence_vec

train_sentence_vec = mydataset(train_dataset)
valid_sentence_vec = mydataset(valid_dataset)
test_sentence_vec = mydataset(test_dataset)

# pickleで保存
print('pickleで保存')
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/train_sentence_vec.pkl', 'wb') as f:
    pickle.dump(train_sentence_vec, f) 
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/valid_sentence_vec.pkl', 'wb') as f:
    pickle.dump(valid_sentence_vec, f) 
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/test_sentence_vec.pkl', 'wb') as f:
    pickle.dump(test_sentence_vec, f) 
