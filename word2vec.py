import re
import MeCab
from gensim.models import Word2Vec
from tqdm import tqdm


# 形態素解析
def tokenize(sentence):
    mecab = MeCab.Tagger('-Owakati')
    text = mecab.parse(sentence)
    return text.strip().split()


# ファイル読み込み
with open('data/text/train.txt') as f:
    train_text = [line.strip() for line in f.readlines()]
    train_text = '\t'.join(train_text)
    train_text = train_text.split('\t')
with open('data/text/valid.txt') as f:
    valid_text = [line.strip() for line in f.readlines()]
    valid_text = '\t'.join(valid_text)
    valid_text = valid_text.split('\t')
with open('data/text/test.txt') as f:
    test_text = [line.strip() for line in f.readlines()]
    test_text = '\t'.join(test_text)
    test_text = test_text.split('\t')

# レシピコーパス  
text = train_text + valid_text + test_text

# 二次元配列で形態素を管理
token_data = []
for sentence in tqdm(text):
    token_data.append(tokenize(sentence))

# word2vecモデル学習（gensim version 3.80）
model = Word2Vec(sentences=token_data,
                 size=300,
                 window=5,
                 min_count=1,
                 workers=8)
# 保存
model.save("data/word2vec.model")
