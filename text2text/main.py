import pickle

import MeCab
import numpy as np
import torch
from scipy import spatial
from gensim.models import Word2Vec
from tqdm import tqdm


# 1000のレシピ文用意
def prepare_recipes():
    # testデータセット読み込み
    with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    text_data = []
    cnt = 1
    for data in test_dataset:
        if cnt > 1000:
            break
        text_data.append(data[0])
        cnt += 1
    return text_data


def text2vec(text):
    # 学習済みword2vec読み込み
    text_model = Word2Vec.load("/mnt/LSTA5/data/tanaka/retrieval/text2image/word2vec.model")
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
    sentence_vec = sentence_sum / len(token_list)
    return sentence_vec

# 文章間の類似度
def sentence_similarity(sen1_vec, sen2_vec):
    return 1 - spatial.distance.cosine(sen1_vec, sen2_vec)

def measurement(target, text_data):
    # 辞書の値からキー抽出
    def get_key_from_value(d, val):
        keys = [k for k, v in d.items() if v == val]
        if keys:
            return keys[0]
        return None

    # 類似度大きい順から10個管理
    similar_dict = {}
    for data in tqdm(text_data, total=len(text_data)):
        if data == target:
            continue
        else:
            similarity = sentence_similarity(text2vec(target), text2vec(data))
            if len(similar_dict) < 10:
                similar_dict[data] = similarity
            else:
                if min(similar_dict.values()) < similarity:
                    key = get_key_from_value(similar_dict, min(similar_dict.values()))
                    del similar_dict[key]
                    similar_dict[data] = similarity
    return similar_dict

def main():
    text_data = prepare_recipes()
    print(sentence_similarity(text2vec(text_data[0]), text2vec(text_data[1])))



if __name__ == '__main__':
    text_data = prepare_recipes()
    similar_dict = measurement('人参を切る', text_data)
    for k, v in similar_dict:
        print(k, v)


