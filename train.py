import pickle
import random

import MeCab
import numpy as np
import torch
from gensim.models import Word2Vec
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm

from model import TripletModel
from matplotlib import pyplot as plt
plt.switch_backend('agg')

# word2vecの学習モデル
word2vec_path = "/mnt/LSTA5/data/tanaka/retrieval/text2image/word2vec.model"
# 学習済みモデル保存パス先頭部分
head_model_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/model/'
# 学習時のloss値保存先
train_losses_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/losses/train_losses.2048.pkl'
valid_losses_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/losses/valid_losses.2048.pkl'
# loss画像保存先
losspng_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/loss_png'

# レシピコーパスで学習したWord2Vec
model = Word2Vec.load(word2vec_path)
# GPU対応
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:1')
# 損失関数
triplet_loss = nn.TripletMarginLoss(margin=1.0)
# 学習させるモデル
triplet_model = TripletModel()


def image2vec(image_net, image_paths):
    # 画像を Tensor に変換
    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # stackはミニバッチに対応できる
    images = torch.stack([
        transformer(Image.open(image_path).convert('RGB'))
        for image_path in image_paths
    ])
    images = images.to(device)
    images = image_net(images)
    return images.cpu()

class MyDataset(Dataset):
    def __init__(self, sentence_vec, image_vec):
        if len(sentence_vec) != len(image_vec):
            print('一致してない')
            exit(0)
        self.data_num = len(sentence_vec)
        self.sentence_vec = sentence_vec
        self.image_vec = image_vec
        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        sentence_vec = self.sentence_vec[idx]
        image_vec = self.image_vec[idx]
        return sentence_vec, image_vec


data_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/'

print('sentence_vec 読み込み中...')
with open(data_path + 'train_sentence_vec.pkl', 'rb') as f:
    train_sentence_vec = pickle.load(f)
with open(data_path + 'valid_sentence_vec.pkl', 'rb') as f:
    valid_sentence_vec = pickle.load(f)
with open(data_path + 'test_sentence_vec.pkl', 'rb') as f:
    test_sentence_vec = pickle.load(f)

print('image_vec 読み込み中...')
with open(data_path + 'train_image_vec.pkl', 'rb') as f:
    train_image_vec = pickle.load(f)
with open(data_path + 'valid_image_vec.pkl', 'rb') as f:
    valid_image_vec = pickle.load(f)
with open(data_path + 'test_image_vec.pkl', 'rb') as f:
    test_image_vec = pickle.load(f)

# PyTorch の Dataset に格納
train_dataset = MyDataset(train_sentence_vec, train_image_vec)
valid_dataset = MyDataset(valid_sentence_vec, valid_image_vec)
test_dataset = MyDataset(test_sentence_vec, test_image_vec)


print('DatasetをDataLoaderにセッティング中...')
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=2048, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=True)


# モデル評価
def eval_net(triplet_model, data_loader, dataset, loss=triplet_loss, device=device):
    triplet_model.eval()
    triplet_model = triplet_model.to(device)
    outputs = []
    for i, (x, y) in enumerate(data_loader):
        # 乱数によりnegative選出
        while True:
            random_idx = random.randint(0, len(dataset)-1)
            negative_text = dataset[random_idx][0]
            negative_image = dataset[random_idx][1]
            if negative_text is not x and negative_image is not y:
                break

        # 画像ベクトルの推測値
        with torch.no_grad():
            # GPU設定
            x = x.to(device)
            y = y.to(device)
            negative_text = negative_text.to(device)
            negative_image = negative_image.to(device)
            # それぞれの次元を512で統一するニューラル
            anchor_text, positive_image = triplet_model(x.float(), y.float())
            negative_text, negative_image = triplet_model(negative_text.float(), negative_image.float())
            positive_text, anchor_image = triplet_model(x.float(), y.float())

        output = loss(anchor_text, positive_image, negative_image) + loss(anchor_image, positive_text, negative_text)
        outputs.append(output.item())

    return sum(outputs) / i 
    

# モデルの学習
def train_net(triplet_model, train_loader, valid_loader, train_dataset, valid_dataset, loss=triplet_loss, n_iter=400, device=device):
    train_losses = []
    valid_losses = []
    optimizer = optim.Adam(triplet_model.parameters())

    for epoch in range(n_iter):
        running_loss = 0.0
        triplet_model = triplet_model.to(device)
        # ネットワーク訓練モード
        triplet_model.train()
        # xxはテキスト、yyは画像
        for i, (xx, yy) in enumerate(train_loader):
            # 乱数によりnegative選出
            while True:
                random_idx = random.randint(0, len(train_dataset)-1)
                negative_text = train_dataset[random_idx][0]
                negative_image = train_dataset[random_idx][1]
                if negative_text is not xx and negative_image is not yy:
                    break
            
            # GPU設定
            xx = xx.to(device)
            yy = yy.to(device)
            negative_text = negative_text.to(device)
            negative_image = negative_image.to(device)

            # それぞれの次元を512で統一するニューラル
            anchor_text, positive_image = triplet_model(xx.float(), yy.float())
            negative_text, negative_image = triplet_model(negative_text.float(), negative_image.float())
            positive_text, anchor_image = triplet_model(xx.float(), yy.float())

            output = loss(anchor_text, positive_image, negative_image) + loss(anchor_image, positive_text, negative_text)
            optimizer.zero_grad()
            output.backward()
            optimizer.step()

            running_loss += output.item()
    
        # 訓練用データでのloss値
        train_losses.append(running_loss / i)
        # 検証用データでのloss値
        pred_valid =  eval_net(triplet_model, valid_loader, valid_dataset, device=device)
        valid_losses.append(pred_valid)
        print('epoch:' +  str(epoch+1), 'train loss:'+ str(train_losses[-1]), 'valid loss:' + str(valid_losses[-1]), flush=True)
        # 学習モデル保存
        if (epoch+1)%50==0:
            # 学習させたモデルの保存パス
            model_path = head_model_path + f'model_epoch{epoch+1}.pth'
            # モデル保存
            torch.save(triplet_model.to('cpu').state_dict(), model_path)
            # loss保存
            with open(train_losses_path, 'wb') as f:
                pickle.dump(train_losses, f) 
            with open(valid_losses_path, 'wb') as f:
                pickle.dump(valid_losses, f) 
            # グラフ描画
            my_plot(train_losses, valid_losses)


def my_plot(train_losses, valid_losses):
    # グラフの描画先の準備
    fig = plt.figure()
    # 画像描画
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    #グラフタイトル
    plt.title('Triplet Margin Loss')
    #グラフの軸
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #グラフの凡例
    plt.legend()
    # グラフ画像保存
    fig.savefig('loss.png')

train_net(triplet_model, train_loader=train_loader, valid_loader=valid_loader, train_dataset=train_dataset, valid_dataset=valid_dataset,  device=device)

# train_net(triplet_model,  train_loader=test_loader, valid_loader=valid_loader, train_dataset=test_dataset, valid_dataset=valid_dataset,  device=device)

