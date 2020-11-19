import os
from gensim.models import FastText
import pickle
import MeCab
import numpy as np
from tqdm import tqdm 



# 1回目（modelが保存されていない時）--------------------------------------------
# gensimからfastTextの学習済み学習済み単語ベクトル表現を利用
# model = FastText.load_fasttext_format('data/cc.ja.300.bin.gz')

# # モデルをストレージに直列化
# with open('data/gensim-vecs.cc.ja.300.bin.pkl', mode='wb') as fp:
#     pickle.dump(model, fp)
# -------------------------------------------------------------------------

# 2回目以降（modelが保存されている時）------------------------------------------
# 非直列化して利用
with open('data/gensim-vecs.cc.ja.300.bin.pkl', mode='rb') as fp:
    model = pickle.load(fp)
# -------------------------------------------------------------------------

# レシピのテキストデータ読み込み
# 訓練用
recipesvec_train = []
total = 101790
with open('data/text/train.txt') as f:
    for line in tqdm(f, total=total):
        recipe = line.split('t')
        recipe_vec = []
        for step in recipe:
            # 形態素解析
            me = MeCab.Tagger("-Owakati")
            me_list = me.parse(step).split()
            # 文章をベクトル化
            sentence = []
            for seg in me_list:
                sentence.append(model.wv[seg])
            sentence = np.array(sentence)
            recipe_vec.append(sentence)
        recipesvec_train.append(recipe_vec)


# step = 'じゃがいもを切ってニンジンを炒めているフライパンに入れ一緒に混ぜ合わせます。'
# # 形態素解析
# me = MeCab.Tagger("-Owakati")
# me_list = me.parse(step).split()
# # 文章をベクトル化
# sentence = []
# for seg in me_list:
#     sentence.append(model.wv[seg])
# # print(sentence)
# # sentence = np.array(sentence)
# # print(type(sentence))
# # print(sentence.shape)
# sentence = np.array(sentence).mean(axis=0)
# print(type(sentence))
# print(sentence.shape)

# # 試験用
# recipesvec_train = []
# total = 101790
# with open('data/text/train.txt') as f:
#     for line in tqdm(f, total=total):
#         recipe = line.split('t')
#         recipe_vec = []
#         for step in recipe:
#             # 形態素解析
#             me = MeCab.Tagger("-Owakati")
#             me_list = me.parse(step).split()
#             # 文章をベクトル化
#             sentence = []
#             for seg in me_list:
#                 sentence.append(model[seg])
#             sentence = np.array(sentence).mean(axis=0)
#             recipe_vec.append(sentence)
#         recipesvec_train.append(recipe_vec)
            





# class Text2VecNet(nn.Module):
#     def __init__(self, vocab_size=vocab_size, embedding_dim=embedding_dim, out_features=50, weights=weights):
#         super().__init__()
#         self.emb = nn.Embedding(vocab_size, embedding_dim)
#         self.emb.weight = nn.Parameter(torch.from_numpy(weights))
#         self.emb.weight.requires_grad = False
#         self.linear = nn.Linear(embedding_dim, out_features)

#     def forward(self, x):
#         x = self.emb(x)
#         x = self.linear(x)
#         return x

# net = Text2VecNet()

