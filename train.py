import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm
import MeCab
import pickle
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from emb_image import OriginalResNet
from gensim.models import Word2Vec


# レシピコーパスで学習したWord2Vec
model = Word2Vec.load("data/word2vec.model")
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
        self.image_vec = []
        for text, image_path in tqdm(dataset, total=self.data_num):
            # 形態素解析
            mecab = MeCab.Tagger("-Owakati")
            token_list = mecab.parse(text).split()
            # 文全体をベクトル化
            sentence = []
            for token in token_list:
                if token in text_model.wv:
                    sentence.append(text_model.wv[token])
                else:
                    continue
            sentence = np.array(sentence).mean(axis=0)
            sentence = torch.from_numpy(sentence).clone()
            self.sentence_vec.append(sentence)

            # 画像のベクトル化
            image = transformer(Image.open(image_path).convert('RGB'))
            self.image_vec.append(image)
    
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        sentence_vec = self.sentence_vec[idx]
        image_vec = self.image_vec[idx]
        return sentence_vec, image_vec



# # デバッグ
# image_path = '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0fffdec2d05716551f0531714536cba6c235a7d0_7.jpg'
# image = Image.open(image_path)
# image_vec = net(transformer(image).unsqueeze(0))
# print(image_vec.shape)

# Datasetを作成
非直列化して利用
with open('data/dataset/train_dataset.bin.pkl', mode='rb') as fp:
    train_dataset = pickle.load(fp)
with open('data/dataset/valid_dataset.bin.pkl', mode='rb') as fp:
    valid_dataset = pickle.load(fp)
with open('data/dataset/test_dataset.bin.pkl', mode='rb') as fp:
    test_dataset = pickle.load(fp)

# path = '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/075409b4165e900458b343462ffb531f7fccc60d_5.jpg'
# image = Image.open(path).convert('RGB')
# image = transformer(image)

# special_dataset = [
#     ['ルッコラは洗って水気を拭き取る。 豚バラは一口大にカットしてみじん切りにしたらパセリ、塩コショウで下味をつける。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_1.jpg'],
#     ['パスタを茹で始める。8分', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_2.jpg'],
#     ['にんにくを温めたら豚バラを炒める。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_3.jpg'],
#     ['豚バラの色が変わったらルッコラを入れて軽く火を通す。 トマト缶を投入。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_4.jpg'],
#     ['しばらくコトコト。パスタの茹で汁と塩で味を調整。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_5.jpg'],
#     ['茹で上がったパスタを投入。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_6.jpg'],
#     ['盛りつけたら完成。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_7.jpg']
# ]

print('pickleで読み込んだデータセットをDatasetに変換中...')
# Datasetに変換
train_dataset = Dataset2Tensor(train_dataset)
valid_dataset = Dataset2Tensor(valid_dataset)
test_dataset = Dataset2Tensor(test_dataset)
# special_dataset = MyDataset(special_dataset)


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
        # 乱数によりnegative選出
        while True:
            random_idx = random.randint(0, len(dataset)-1)
            negative = dataset[random_idx][1].unsqueeze(0)
            
            if negative is not y:
                break
        # GPU設定
        x = x.to(device)
        y = y.to(device)
        negative = negative.to(device)
        # 画像ベクトルの推測値
        with torch.no_grad():
            y = image_net(y)
            negative = image_net(negative)

        anchor = x
        positive = y 
        
        output = loss(anchor, positive, negative)
        outputs.append(output.item())

    return sum(outputs) / i 

        
# # eval_net(image_net, test_loader, test_dataset)
        


# モデルの学習
def train_net(image_net, train_loader, valid_loader, train_dataset, valid_dataset, only_fc=True, loss=triplet_loss, n_iter=20, device='cpu'):
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
            # 乱数によりnegative選出
            while True:
                random_idx = random.randint(0, len(train_dataset)-1)
                negative = train_dataset[random_idx][1].unsqueeze(0)
                if negative is not yy:
                    break
            # GPU設定
            xx = xx.to(device)
            yy = yy.to(device)
            negative = negative.to(device)

            # 画像ベクトルの推測値
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
        if (epoch+1)%5==0:
            # 学習させたモデルの保存パス
            model_path = f'data/model/model_{epoch+1}.pth'
            # モデル保存
            torch.save(image_net.state_dict(), model_path)

    # グラフ描画
    # my_plot(np.linspace(1, n_iter, n_iter).astype(int), train_losses, valid_losses)


def my_plot(epochs, train_losses, valid_losses):
    # グラフの描画先の準備
    fig = plt.figure()
    # グラフ描画
    plt.plot(epochs, train_losses, color = 'red')
    plt.plot(epochs, valid_losses, color = 'blue')
    # グラフをファイルに保存する
    fig.savefig("img.png")

train_net(image_net, train_loader=special_loader, valid_loader=special_loader, train_dataset=special_dataset, valid_dataset=special_dataset,  device=device)

