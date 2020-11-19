import re
import MeCab



# ファイル読み込み
with open('data/text/train.txt') as f:
    train_text = [line.strip() for line in f.readlines()]
    train_text = [re.sub('\t', '', line) for line in train_text]
with open('data/text/valid.txt') as f:
    valid_text = [line.strip() for line in f.readlines()]
    valid_text = [re.sub('\t', '', line) for line in valid_text]
with open('data/text/test.txt') as f:
    test_text = [line.strip() for line in f.readlines()]
    test_text = [re.sub('\t', '', line) for line in test_text]

text = train_text + valid_text + test_text

me = MeCab.Tagger()
