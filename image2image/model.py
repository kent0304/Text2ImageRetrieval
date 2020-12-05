# OriginalResnetは2048次元の特徴量を出力する学習済みモデル
# TripletModel300次元のテキストと2048次元の画像を512次元にそれぞれ合わせるモデル
import torch
from torch import nn, optim
import torch.nn.functional as F
import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models
from PIL import Image


# モデル定義
class OriginalResNet(nn.Module):
    def __init__(self, metric_dim=300):
        super().__init__()
        # 事前学習済みのresnet50をダウンロード
        resnet = models.resnet50(pretrained=True) 
        # 最終層以外はそのまま利用
        modules = list(resnet.children())[:-1]
        self.model = nn.Sequential(*modules)
        # 全てのパラメータを微分対象外にする
        for params in self.model.parameters():
            params.requires_grad = False
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x

class TripletModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_fc1 = nn.Linear(2048, 1536)
        self.image_fc2 = nn.Linear(1536, 1024)
        self.image_fc3 = nn.Linear(1024, 512)
        self.text_fc1 = nn.Linear(300, 400)
        self.text_fc2 = nn.Linear(400, 450)
        self.text_fc3 = nn.Linear(450, 512)

    def forward(self, text, image):
        text = F.relu(self.text_fc1(text))
        text = F.relu(self.text_fc2(text))
        text_vec = self.text_fc3(text)

        image = F.relu(self.image_fc1(image))
        image = F.relu(self.image_fc2(image))
        image_vec = self.image_fc3(image)

        return text_vec, image_vec
