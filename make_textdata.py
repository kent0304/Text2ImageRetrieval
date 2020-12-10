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

# レシピコーパスで学習したWord2Vec
text_model_300 = Word2Vec.load("/mnt/LSTA5/data/tanaka/retrieval/text2image/word2vec_300.model")

def mydataset(dataset, text_model):
    data_num = len(dataset)
    sentence_vec = []
    for data in tqdm(dataset, total=data_num):
        text = data[0]
        # 形態素解析
        mecab = MeCab.Tagger("-Owakati")
        token_list = mecab.parse(text).split()
        stopwords = ['を', '。', 'に', '、', 'て', 'の', 'で', 'ます', 'し', 'は', 'が', 'た', 'と', 'たら', '分', 'です', 'も', 'お', '！', '!', '１', '・', '（', '）', 'さ', 'まで', 'から', '1', '０', '♪', '２', '～', 'せ', '2', '３', '☆', 'ば', '５', '3', '(', ')']
        # 文全体をベクトル化
        sentence_sum = np.zeros(text_model.wv.vectors.shape[1], )
        cnt = 0
        for token in token_list:
            if token in stopwords:
                continue
            if token in text_model.wv:
                sentence_sum += text_model.wv[token]
                cnt += 1
            else:
                continue
        if cnt == 0:
            cnt = 1
        sentence = sentence_sum / cnt
        sentence = torch.from_numpy(sentence).clone()
        sentence_vec.append(sentence)
    return sentence_vec

train_sentence_vec = mydataset(train_dataset, text_model=text_model_300)
valid_sentence_vec = mydataset(valid_dataset, text_model=text_model_300)
test_sentence_vec = mydataset(test_dataset, text_model=text_model_300)

# pickleで保存
print('pickleで保存')
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/word2vec300/train_sentence_vec.pkl', 'wb') as f:
    pickle.dump(train_sentence_vec, f) 
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/word2vec300/valid_sentence_vec.pkl', 'wb') as f:
    pickle.dump(valid_sentence_vec, f) 
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/word2vec300/test_sentence_vec.pkl', 'wb') as f:
    pickle.dump(test_sentence_vec, f) 