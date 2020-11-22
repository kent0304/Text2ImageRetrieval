import torch
from torch import nn, optim
import numpy as np
import tqdm
import MeCab
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from emb_image import OriginalResNet
from gensim.models import Word2Vec

# DatasetをTensorに変換
def Dataset2Tensor(dataset, net=net, transformer=transformer):
    new_dataset = []
    for text, image_path in dataset:
        # テキストのベクトル化
        # 形態素解析
        mecab = MeCab.Tagger("-Owakati")
        token_list = mecab.parse(step).split()
        # 文全体をベクトル化
        sentence = []
        for token in token_list:
            if token in model.wv:
                sentence.append(model.wv[token])
            else:
                continue
        sentence = np.array(sentence).mean(axis=0)
        sentence = torch.from_numpy(sentence).clone()

        # 画像のベクトル化
        image = Image.open(image_path)
        image_vec = net(transformer(image).unsqueeze(0))

        new_dataset.append([sentence])
    return new_dataset


# レシピコーパスで学習したWord2Vec
model = Word2Vec.load("data/word2vec.model")
# GPU対応
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 損失関数
triplet_loss = nn.TripletMarginLoss()
# netにはモデルを代入
net = OriginalResNet()
# 画像を Tensor に変換
transformer = transforms.Compose([
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# # デバッグ
# image_path = '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0fffdec2d05716551f0531714536cba6c235a7d0_7.jpg'
# image = Image.open(image_path)
# image_vec = net(transformer(image).unsqueeze(0))
# print(image_vec.shape)


# Datasetを作成
# 非直列化して利用
with open('data/dataset/train_dataset.bin.pkl', mode='rb') as fp:
    train_dataset = pickle.load(fp)
with open('data/dataset/valid_dataset.bin.pkl', mode='rb') as fp:
    valid_dataset = pickle.load(fp)
with open('data/dataset/test_dataset.bin.pkl', mode='rb') as fp:
    test_dataset = pickle.load(fp)

# Tensorに変換
train_dataset = Dataset2Tensor(train_dataset)
valid_dataset = Dataset2Tensor(valid_dataset)
test_dataset = Dataset2Tensor(test_dataset)

# DataLoaderを作成
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)


# モデルの学習
def train_net(net, train_loader, valid_loader, only_fc=True, loss_fn=loss_fn, n_iter=10, device='cpu'):
    train_losses = []
    valid_losses = []
    net = net.to(device)
    # 最後の線形層のパラメータのみを optimizer に渡す
    if only_fc:
        optimizer = optim.Adam(net.fc.parameters())
    else:
        optiizer = optim.Adam(net.parameters())

    for epoch in range(n_iter):
        running_loss = 0.0
        # ネットワーク訓練モード
        net.train()

        # xxはテキストanchor、yyはpositive画像
        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            xx = xx.to(device)
            yy = yy.to(device)

            anchor = xx
            positive = yy 
            negative = torch.randn(100, 128, requires_grad=True)
            output = triplet_loss(anchor, positive, negative)
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            running_loss += output.item()
        
        train_losses.append(running_loss / i)
        print(epoch, train_losses[-1], flush=True)


