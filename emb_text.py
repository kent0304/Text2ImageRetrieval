import os
import pickle
import MeCab
import numpy as np
from tqdm import tqdm 
from gensim.models import Word2Vec


# レシピコーパスで学習したWord2Vec
model = Word2Vec.load("data/word2vec.model")

# FastText使う場合
# 1回目（modelが保存されていない時）--------------------------------------------
# gensimからfastTextの学習済み学習済み単語ベクトル表現を利用
# model = FastText.load_fasttext_format('data/cc.ja.300.bin.gz')

# # モデルをストレージに直列化
# with open('data/gensim-vecs.cc.ja.300.bin.pkl', mode='wb') as fp:
#     pickle.dump(model, fp)
# 2回目以降（modelが保存されている時）------------------------------------------
# 非直列化して利用
# with open('data/gensim-vecs.cc.ja.300.bin.pkl', mode='rb') as fp:
#     model = pickle.load(fp)
# -------------------------------------------------------------------------

# レシピのテキストデータ読み込み
# 訓練用
recipesvec_train = []
total = 101790
with open('data/text/train.txt') as f:
    for line in tqdm(f, total=total):
        recipe = line.split('\t')
        for step in recipe:
            # 形態素解析
            mecab = MeCab.Tagger("-Owakati")
            token_list = mecab.parse(step).split()
            # 文章をベクトル化
            sentence = []
            for token in token_list:
                if token in model.wv:
                    sentence.append(model.wv[token])
                else:
                    continue
            sentence = np.array(sentence).mean(axis=0)
            recipesvec_train.append(sentence)

# 検証用
recipesvec_valid = []
total = 11324
with open('data/text/valid.txt') as f:
    for line in tqdm(f, total=total):
        recipe = line.split('\t')
        for step in recipe:
            # 形態素解析
            mecab = MeCab.Tagger("-Owakati")
            token_list = mecab.parse(step).split()
            # 文章をベクトル化
            sentence = []
            for token in token_list:
                if token in model.wv:
                    sentence.append(model.wv[token])
                else:
                    continue
            sentence = np.array(sentence).mean(axis=0)
            recipesvec_valid.append(sentence)
   
# テスト用
recipesvec_test = []
total = 12546
with open('data/text/test.txt') as f:
    for line in tqdm(f, total=total):
        recipe = line.split('\t')
        # recipe_vec = []
        for step in recipe:
            # 形態素解析
            mecab = MeCab.Tagger("-Owakati")
            token_list = mecab.parse(step).split()
            # 文章をベクトル化
            sentence = []
            for token in token_list:
                if token in model.wv:
                    sentence.append(model.wv[token])
                else:
                    continue
            sentence = np.array(sentence).mean(axis=0)
            recipesvec_test.append(sentence)

# ベクトル化したテキストのnumpyを保存            
np.save('data/textvec/recipesvec_train',recipesvec_train)
np.save('data/textvec/recipesvec_valid',recipesvec_valid)
np.save('data/textvec/recipesvec_test',recipesvec_test)

    
# print(len(recipesvec_test))
# print(type(recipesvec_test))
# recipesvec_test = np.array(recipesvec_test)
# print(len(recipesvec_test))
# print(type(recipesvec_test))
# print(recipesvec_test.shape)
# print(recipesvec_test[0])
# print(type(recipesvec_test[0]))
# print(recipesvec_test[0].shape)








# # デバッグ
# step = 'ﾓﾝｽﾀｰﾎﾞｰﾙを作ります。ﾊﾑの上にｽﾗｲｽﾁｰｽﾞを乗せます。	ｶﾆｶﾏは赤い部分と白い部分に分け赤い部分をﾁｰｽﾞの上に乗せます。	丸い型を用意してｶﾆｶﾏの直線がだいたい真中になる様に合わせて型を抜きます。	こんな感じになります。	海苔を画像の様に直線と丸に切り貼ります。ｽﾗｲｽﾁｰｽﾞを適当な丸に切り乗せて完成！ﾁｰｽﾞは爪楊枝で簡単に切れます。	ﾋﾟﾁｭｰを作ります。ｵｰﾌﾞﾝｼｰﾄにﾋﾟﾁｭｰを写したのと輪郭だけ写したのを用意します。	輪郭の方はﾊｻﾐで切り取っておきソレをﾁｪﾀﾞｰﾁｰｽﾞの上に乗せて爪楊枝で切ります。	更にハムの上に置きます。	最初に写したﾋﾟﾁｭｰを海苔の上に置きｽﾞﾚない様に黒い部分を切り抜きます。	切った海苔を配置します。	目はｽﾗｲｽﾁｰｽﾞで丸く作りﾏﾖﾈｰｽﾞで貼り付けます。	カニカマでホッペを作りﾏﾖﾈｰｽﾞで貼り付けて周りのハムを丁寧に切り取り出来上がり。'
# # 形態素解析
# me = MeCab.Tagger("-Owakati")
# me_list = me.parse(step).split()
# # 文章をベクトル化
# sentence = []
# print(me_list)
# for token in me_list:
#     if token in model.wv:
#         sentence.append(model.wv[token])
#     else:
#         continue
# # print(sentence)
# # sentence = np.array(sentence)
# # print(type(sentence))
# # print(sentence.shape)
# sentence = np.array(sentence).mean(axis=0)
# print(type(sentence))
# print(sentence.shape)
# print(sentence)




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

