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
from torch.utils.data import Dataset, DataLoader, TensorDataset
from model import TripletModel
from gensim.models import Word2Vec

# special_dataset = [
#     ['ルッコラは洗って水気を拭き取る。 豚バラは一口大にカットしてみじん切りにしたらパセリ、塩コショウで下味をつける。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_1.jpg'],
#     ['パスタを茹で始める。8分', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_2.jpg'],
#     ['にんにくを温めたら豚バラを炒める。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_3.jpg'],
#     ['豚バラの色が変わったらルッコラを入れて軽く火を通す。 トマト缶を投入。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_4.jpg'],
#     ['しばらくコトコト。パスタの茹で汁と塩で味を調整。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_5.jpg'],
#     ['茹で上がったパスタを投入。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_6.jpg'],
#     ['盛りつけたら完成。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_7.jpg']
# ]


# レシピコーパスで学習したWord2Vec
model = Word2Vec.load("/mnt/LSTA5/data/tanaka/retrieval/text2image/word2vec.model")
# GPU対応
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:0')
# 損失関数
triplet_loss = nn.TripletMarginLoss()
# 学習させるモデル
triplet_model = TripletModel()
# 画像を Tensor に変換
transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transforms.RandomCrop(256),
])


# ファイルの読み込み
def read_file(file):
    recipe_text = []
    # 平仮名片仮名英数字を検出する正規表現
    p = '[\u3041-\u309F]+|[\u30A1-\u30F4]+|[a-zA-Z\-[\]]+'
    with open(file) as f:
        steps = f.readlines()
    for step in steps:
        if step == '':
            if '\t' in step:
                step = step.split('\t')[1].rstrip()
                recipe_text.append(step)
        obj = re.search(p, step)
        if obj:
            # 先頭文字が番号かどうか正規表現で確認
            m = re.match(r'\d{2}.*', step)
            # 先頭文字が番号でなければ前の行から続き
            if m is None:
                if len(recipe_text)>0:
                    recipe_text[-1] += step.strip()
                else:
                    continue
            # 先頭文字が番号の場合
            else: 
                if '\t' in step:
                    step = step.split('\t')[1].rstrip()
                    recipe_text.append(step)
        else:
            continue
    return recipe_text

# フォルダ内の画像をジェネレータで出力
def listup_imgs(path):
    return list(os.path.abspath(p) for p in glob.glob(path))

# train, valid, test それぞれのレシピID（ハッシュ列取得）
def get_hash(data):
    with open(f'/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/recipe_ids/{data}.txt') as f:
        return [line.strip() for line in f.readlines()]

def make_dataset(recipe_data):
    # 二次元配列で返す
    dataset = []
    # recipe_ids（ハッシュ列） とそれがどのディレクトリに保存されてるか記してるファイル読み込み
    with open('/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/cookpad_nii/main/file_relation_list.txt') as f:
        file_relation_list = [line.strip().split('\t') for line in f.readlines()]

    # recipeid と 保存されているディレクトリ のペアの dictionary
    relation_dict = {k: v for (k,v) in file_relation_list}

    print('確認しながらテキストと画像のpathを回収していく〜')
    for recipe in tqdm(recipe_data, total=len(recipe_data)):
        # テキスト
        recipe_dir = relation_dict[recipe]
        # 3桁いないのディレクトリ名は前にを足してパス変更
        if len(recipe_dir) < 4:
            num = 4 - len(recipe_dir)
            recipe_dir = '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/cookpad_nii/main/' + '0' * num + recipe_dir + '/' + recipe + '/' + 'step_memos.txt'
        else:
            continue
        # テキストファイル読み込み
        recipe_text = read_file(recipe_dir)
        # 画像
        head = recipe[0]
        path = '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/' + head + '/' + recipe + '*'
        # globは返すリストの順序を保証しないのでnatsortedで辞書順ソート
        img_paths = natsorted(glob.glob(path))
        if len(recipe_text) != len(img_paths):
            continue

        # 画像は余分にあるのでテキスト基準で
        for i in range(len(recipe_text)):
            dataset.append([recipe_text[i], img_paths[i]])

    return dataset

# class MyDataset(Dataset):
#     def __init__(self, dataset, text_model=model, transformer=transformer):
#         self.data_num = len(dataset)
#         self.sentence_vec = []
#         self.image_paths = []
#         for text, image_path in tqdm(dataset, total=self.data_num):
#             if text=='' or image_path=='':
#                 continue
#             # 形態素解析
#             mecab = MeCab.Tagger("-Owakati")
#             token_list = mecab.parse(text).split()
#             # 文全体をベクトル化
#             sentence_sum = np.zeros(text_model.wv.vectors.shape[1], )
#             for token in token_list:
#                 if token in text_model.wv:
#                     sentence_sum += text_model.wv[token]
#                 else:
#                     continue
#             sentence = sentence_sum / len(token_list)
#             sentence = torch.from_numpy(sentence).clone()
#             self.sentence_vec.append(sentence)

#             # 画像
            
#             self.image_paths.append(image_path)
        
#     def __len__(self):
#         return self.data_num

#     def __getitem__(self, idx):
#         sentence_vec = self.sentence_vec[idx]
#         image_path = self.image_paths[idx]
#         # 画像のベクトル化
#         # image_vec = image2vec(image_path)
#         return sentence_vec, image_path

# def image2vec(image_path):
#     # ブログ参考
#     image_net = models.resnet50(pretrained=True)
#     image_net.fc = nn.Identity()

#     # 画像を Tensor に変換
#     transformer = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     # 画像のベクトル化
#     image_net.eval()
#     image_net = image_net.to(device)
#     image = transformer(Image.open(image_path).convert('RGB')).unsqueeze(0)
#     image = image.to(device)
#     image = image_net(image)
#     return image.to('cpu').flatten()




 
print('train/valid/testに分けられたハッシュ列取得...')
recipe_train = get_hash('train')
recipe_valid = get_hash('val')
recipe_test = get_hash('test')

print('データセットから二次元配列に生テキストと画像パス保管中...')
train_dataset = make_dataset(recipe_train)
valid_dataset = make_dataset(recipe_valid)
test_dataset = make_dataset(recipe_test)

# pickleで保存
print('pickleで保存')
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/train_dataset.pkl', 'wb') as f:
    pickle.dump(train_dataset, f) 
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/valid_dataset.pkl', 'wb') as f:
    pickle.dump(valid_dataset, f) 
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/test_dataset.pkl', 'wb') as f:
    pickle.dump(test_dataset, f) 

# # pickleで読み込み
# with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/train_dataset.pkl', 'rb') as f:
#     train_dataset = pickle.load(f)
# with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/valid_dataset.pkl', 'rb') as f:
#     valid_dataset = pickle.load(f)
# with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/test_dataset.pkl', 'rb') as f:
#     test_dataset = pickle.load(f)


# print('データセットをDatasetに変換中...')
# train_dataset = MyDataset(train_dataset)
# valid_dataset = MyDataset(valid_dataset)
# test_dataset = MyDataset(test_dataset)
# special_dataset = MyDataset(special_dataset)
# print(special_dataset[0])
# print(special_dataset[0][0].shape)
# print(special_dataset[0][1].shape)




# # # pickleで保存
# print('pickleで保存')
# with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/train_dataset.pkl', 'wb') as f:
#     pickle.dump(train_dataset, f) 
# with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/valid_dataset.pkl', 'wb') as f:
#     pickle.dump(valid_dataset, f) 
# with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/test_dataset.pkl', 'wb') as f:
#     pickle.dump(test_dataset, f) 
# with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/special_dataset.pkl', 'wb') as f:
#     pickle.dump(special_dataset, f) 