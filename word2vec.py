import re
import MeCab
from gensim.models import Word2Vec
from tqdm import tqdm
import itertools
from collections import Counter


# 形態素解析
def tokenize(sentence):
    mecab = MeCab.Tagger('-Owakati')
    text = mecab.parse(sentence)
    return text.strip().split()


# ファイル読み込み
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/text/train.txt') as f:
    train_text = [line.strip() for line in f.readlines()]
    train_text = '\t'.join(train_text)
    train_text = train_text.split('\t')
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/text/valid.txt') as f:
    valid_text = [line.strip() for line in f.readlines()]
    valid_text = '\t'.join(valid_text)
    valid_text = valid_text.split('\t')
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/text/test.txt') as f:
    test_text = [line.strip() for line in f.readlines()]
    test_text = '\t'.join(test_text)
    test_text = test_text.split('\t')

# レシピコーパス  
text = train_text + valid_text + test_text

# 二次元配列で形態素を管d理
token_data = []
for sentence in tqdm(text):
    token_data.append(tokenize(sentence))

# 頻出ワード確認
flatten_list = list(itertools.chain.from_iterable(token_data))
fdist = Counter(flatten_list)
stop_words = fdist.most_common(n=100)

# word2vecモデル学習（gensim version 3.80）
model_300 = Word2Vec(sentences=token_data,
                 size=300,
                 window=5,
                 min_count=1,
                 iter=50,
                 workers=8)
# 保存
model_300.save("/mnt/LSTA5/data/tanaka/retrieval/text2image/word2vec_300.model")