import pickle
import random

import MeCab
import numpy as np
import torch
from gensim.models import Word2Vec
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model import OriginalResNet, TripletModel
from matplotlib import pyplot as plt
plt.switch_backend('agg')

# word2vecの学習モデル
word2vec_path = "/mnt/LSTA5/data/tanaka/retrieval/text2image/word2vec.model"
# pytorchのデータセット二次元配列
train_dataset_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/train_dataset.pkl'
valid_dataset_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/valid_dataset.pkl'
test_dataset_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/test_dataset.pkl'
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 損失関数
triplet_loss = nn.TripletMarginLoss(margin=0.1)
# netにはモデルを代入
image_net = OriginalResNet()
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

# pickleで読み込み
print('pickelでdataset読み込み中...')
with open(train_dataset_path, 'rb') as f:
    train_dataset = pickle.load(f)
with open(valid_dataset_path, 'rb') as f:
    valid_dataset = pickle.load(f)
with open(test_dataset_path, 'rb') as f:
    test_dataset = pickle.load(f)
# special_dataset = [
#     ['ルッコラは洗って水気を拭き取る。 豚バラは一口大にカットしてみじん切りにしたらパセリ、塩コショウで下味をつける。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_1.jpg'],
#     ['パスタを茹で始める。8分', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_2.jpg'],
#     ['にんにくを温めたら豚バラを炒める。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_3.jpg'],
#     ['豚バラの色が変わったらルッコラを入れて軽く火を通す。 トマト缶を投入。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_4.jpg'],
#     ['しばらくコトコト。パスタの茹で汁と塩で味を調整。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_5.jpg'],
#     ['茹で上がったパスタを投入。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_6.jpg'],
#     ['盛りつけたら完成。', '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0c6aad2feba89eb88219e69c2b9b15c8a1d62045_7.jpg']
# ]

print('DatasetをDataLoaderにセッティング中...')
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=2048, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
# special_loader = DataLoader(special_dataset, batch_size=1, shuffle=True)

# モデル評価
def eval_net(triplet_model,image_net, data_loader, dataset, loss=triplet_loss, device="cpu"):
    triplet_model.eval()
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
            # 画像のベクトル化
            y = image_net(y)
            negative = image_net(negative)
            # 次元合わせ
            anchor, positive = triplet_model(x.float(), y.float())
            _, negative = triplet_model(x.float(), negative.float())
        
        output = loss(anchor, positive, negative)
        outputs.append(output.item())

    return sum(outputs) / i 
    

# モデルの学習
def train_net(triplet_model, image_net, train_loader, valid_loader, train_dataset, valid_dataset, loss=triplet_loss, n_iter=400, device='cpu'):
    train_losses = []
    valid_losses = []
    optimizer = optim.Adam(triplet_model.parameters())
    triplet_model = triplet_model.to(device)
    image_net = image_net.to(device)

    for epoch in range(n_iter):
        running_loss = 0.0
        # ネットワーク訓練モード
        triplet_model.train()
        # xxはテキストanchor、yyはpositive画像
        for i, (xx, yy) in tqdm(enumerate(train_loader), total=len(train_loader)):
            yy = yy[0]
            # 乱数によりnegative選出
            while True:
                random_idx = random.randint(0, len(train_dataset)-1)
                negative = train_dataset[random_idx][1]
                if negative is not yy:
                    break
            
            # 画像のベクトル化
            yy = transformer(Image.open(yy).convert('RGB')).unsqueeze(0)
            negative = transformer(Image.open(negative).convert('RGB')).unsqueeze(0)
            # GPU設定
            xx = xx.to(device)
            yy = yy.to(device)
            negative = negative.to(device)
            # 画像のベクトル化（2048次元）
            yy = image_net(yy)
            negative = image_net(negative)
            # それぞれの次元を512で統一するニューラル
            anchor, positive = triplet_model(xx.float(), yy.float())
            _, negative = triplet_model(xx.float(), negative.float())

            output = loss(anchor, positive, negative)
            optimizer.zero_grad()
            output.backward()
            optimizer.step()

            running_loss += output.item()
        
        # 訓練用データでのloss値
        train_losses.append(running_loss / i)
        # 検証用データでのloss値
        pred_valid =  eval_net(triplet_model, image_net, valid_loader, valid_dataset, device=device)
        valid_losses.append(pred_valid)
        print('epoch:' +  str(epoch+1), ', train loss:'+ str(train_losses[-1]), ', valid loss:' + str(valid_losses[-1]), flush=True)
        # 学習モデル保存
        if (epoch+1)%50==0:
            # 学習させたモデルの保存パス
            model_path = head_model_path + f'model_2048_{epoch+1}.pth'
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
    # 値用意
    #t_losses = [1.01, 1.00, 0.99, 0.97, 0.97, 0.96, 0.95, 0.93, 0.92, 0.91, 0.90, 0.87, 0.85, 0.83]
    #v_losses = [1.01, 1.00, 0.99, 0.97, 0.97, 0.99, 0.95, 0.93, 0.96, 0.97, 0.99, 0.99, 0.95, 0.93]
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

train_net(triplet_model, image_net, train_loader=train_loader, valid_loader=valid_loader, train_dataset=train_dataset, valid_dataset=valid_dataset,  device=device)

# train_net(triplet_model, image_net, train_loader=special_loader, valid_loader=special_loader, train_dataset=special_dataset, valid_dataset=special_dataset,  device=device)

