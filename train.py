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
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from natsort import natsorted
from PIL import Image
from multiprocessing import Pool
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from emb_image import OriginalResNet
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
model = Word2Vec.load("/mnt/LSTA5/data/tanaka/data/word2vec.model")
# GPU対応
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 損失関数
triplet_loss = nn.TripletMarginLoss()
# netにはモデルを代入
image_net = OriginalResNet()
# 画像を Tensor に変換
transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transforms.RandomCrop(256),
])

class MyDataset(Dataset):
    def __init__(self, dataset, text_model=model, transformer=transformer):
        self.data_num = len(dataset)
        self.sentence_vec = []
        self.image_paths = []
        for text, image_path in tqdm(dataset, total=self.data_num):
            if text=='' or image_path=='':
                continue
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
            self.sentence_vec.append(sentence)

            # 画像のベクトル化
            # image = transformer(Image.open(image_path).convert('RGB'))
            self.image_paths.append(image_path)
    
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        sentence_vec = self.sentence_vec[idx]
        image_path = self.image_paths[idx]
        return sentence_vec, image_path


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
            # print('あかん')
            # print('ハッシュ値：', recipe)
            # print('テキストのパスは')
            # print(recipe_dir)
            # print('テキスト数は')
            # print(len(recipe_text))
            # print('テキストは')
            # print(recipe_text)
            # print('画像のパスは')
            # print(img_paths)

        # 画像は余分にあるのでテキスト基準で
        for i in range(len(recipe_text)):
            dataset.append([recipe_text[i], img_paths[i]])

    return dataset
 
print('train/valid/testに分けられたハッシュ列取得...')
recipe_train = get_hash('train')
recipe_valid = get_hash('val')
recipe_test = get_hash('test')

print('データセットから二次元配列に生テキストと画像パス保管中...')
# train_dataset = make_dataset(recipe_train)
# valid_dataset = make_dataset(recipe_valid)
# test_dataset = make_dataset(recipe_test)

# # pickleで保存
# print('pickleで保存')
# with open('/mnt/LSTA5/data/tanaka/data/dataset/train_dataset.pkl', 'wb') as f:
#     pickle.dump(train_dataset, f) 
# with open('/mnt/LSTA5/data/tanaka/data/dataset/valid_dataset.pkl', 'wb') as f:
#     pickle.dump(valid_dataset, f) 
# with open('/mnt/LSTA5/data/tanaka/data/dataset/test_dataset.pkl', 'wb') as f:
#     pickle.dump(test_dataset, f) 

# pickleで読み込み
with open('/mnt/LSTA5/data/tanaka/data/dataset/train_dataset.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
with open('/mnt/LSTA5/data/tanaka/data/dataset/valid_dataset.pkl', 'rb') as f:
    valid_dataset = pickle.load(f)
with open('/mnt/LSTA5/data/tanaka/data/dataset/test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)


print('データセットをDatasetに変換中...')
# Datasetに変換
train_dataset = MyDataset(train_dataset)
valid_dataset = MyDataset(valid_dataset)
test_dataset = MyDataset(test_dataset)
# special_dataset = MyDataset(special_dataset)

# pickleで保存
print('pickleで保存')
with open('/mnt/LSTA5/data/tanaka/data/torch_dataset/train_dataset.pkl', 'wb') as f:
    pickle.dump(train_dataset, f) 
with open('/mnt/LSTA5/data/tanaka/data/torch_dataset/valid_dataset.pkl', 'wb') as f:
    pickle.dump(valid_dataset, f) 
with open('/mnt/LSTA5/data/tanaka/data/torch_dataset/test_dataset.pkl', 'wb') as f:
    pickle.dump(test_dataset, f) 


print('DatasetをDataLoaderにセッティング中...')
# DataLoaderを作成
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# special_loader = DataLoader(special_dataset, batch_size=1, shuffle=True)

# モデル評価
def eval_net(image_net, data_loader, dataset, loss=triplet_loss, device="cpu"):
    image_net.eval()
    outputs = []
    for i, (x, y) in enumerate(data_loader):
        y = y[0]
        # 乱数によりnegative選出
        while True:
            random_idx = random.randint(0, len(dataset)-1)
            negative = dataset[random_idx][1]
            if negative is not y:
                break

        # 画像ベクトルの推測値
        with torch.no_grad():
            # 画像ベクトル化
            y = transformer(Image.open(y).convert('RGB')).unsqueeze(0)
            negative = transformer(Image.open(negative).convert('RGB')).unsqueeze(0)
            # GPU設定
            x = x.to(device)
            y = y.to(device)
            negative = negative.to(device)
            # 推論
            y = image_net(y)
            negative = image_net(negative)


        anchor = x
        positive = y 
        
        output = loss(anchor, positive, negative)
        outputs.append(output.item())

    return sum(outputs) / i 

        
# # eval_net(image_net, test_loader, test_dataset)
        


# モデルの学習
def train_net(image_net, train_loader, valid_loader, train_dataset, valid_dataset, only_fc=True, loss=triplet_loss, n_iter=400, device='cpu'):
    train_losses = []
    valid_losses = []
    image_net = image_net.to(device)
    # 最後の線形層のパラメータのみを optimizer に渡す
    if only_fc:
        optimizer = optim.Adam(image_net.fc.parameters())
    else:
        optiizer = optim.Adam(image_net.parameters())

    for epoch in range(n_iter):
        running_loss = 0.0
        # ネットワーク訓練モード
        image_net.train()
        # xxはテキストanchor、yyはpositive画像
        for i, (xx, yy) in tqdm(enumerate(train_loader), total=len(train_loader)):
            yy = yy[0]
            # 乱数によりnegative選出
            while True:
                random_idx = random.randint(0, len(train_dataset)-1)
                negative = train_dataset[random_idx][1]
                if negative is not yy:
                    break
            
            # 画像ベクトルの推測値
            # 画像のベクトル化
            yy = transformer(Image.open(yy).convert('RGB')).unsqueeze(0)
            negative = transformer(Image.open(negative).convert('RGB')).unsqueeze(0)
            # GPU設定
            xx = xx.to(device)
            yy = yy.to(device)
            negative = negative.to(device)
            # 推論
            yy = image_net(yy)
            negative = image_net(negative)

            anchor = xx
            positive = yy 
            output = loss(anchor, positive, negative)
            optimizer.zero_grad()
            output.backward()
            optimizer.step()

            running_loss += output.item()
        
        # 訓練用データでのloss値
        train_losses.append(running_loss / i)
        # 検証用データでのloss値
        pred_valid =  eval_net(image_net, valid_loader, valid_dataset, device=device)
        valid_losses.append(pred_valid)
        print('epoch:' +  str(epoch+1), ', train loss:'+ str(train_losses[-1]), ', valid loss:' + str(valid_losses[-1]), flush=True)
        # 学習モデル保存
        if (epoch+1)%50==0:
            # 学習させたモデルの保存パス
            model_path = f'/mnt/LSTA5/data/tanaka/data/model/model_{epoch+1}.pth'
            # モデル保存
            torch.save(image_net.to('cpu').state_dict(), model_path)
        # loss保存
        with open('/mnt/LSTA5/data/tanaka/data/losses/train_losses.pkl', 'wb') as f:
            pickle.dump(train_losses, f) 
        with open('/mnt/LSTA5/data/tanaka/data/losses/valid_losses.pkl', 'wb') as f:
            pickle.dump(valid_losses, f) 


#     # グラフ描画
#     # my_plot(np.linspace(1, n_iter, n_iter).astype(int), train_losses, valid_losses)


# def my_plot(epochs, train_losses, valid_losses):
#     # グラフの描画先の準備
#     fig = plt.figure()
#     # グラフ描画
#     plt.plot(epochs, train_losses, color = 'red')
#     plt.plot(epochs, valid_losses, color = 'blue')
#     # グラフをファイルに保存する
#     fig.savefig("img.png")

train_net(image_net, train_loader=train_loader, valid_loader=valid_loader, train_dataset=train_dataset, valid_dataset=valid_dataset,  device=device)

# train_net(image_net, train_loader=special_loader, valid_loader=special_loader, train_dataset=special_dataset, valid_dataset=special_dataset,  device=device)

