import torch
from torch import nn, optim
import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models
from PIL import Image


# 入力画像のpath
path = '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0fffdec2d05716551f0531714536cba6c235a7d0_7.jpg'

image = Image.open(path)
# Tensor に変換
transformer = transforms.Compose([
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# モデル定義
class OriginalResNet(nn.Module):
    def __init__(self, metric_dim=300):
        super().__init__()
        # 事前学習済みのresnet50をダウンロード
        resnet = models.resnet50(pretrained=True)   
        # 全てのパラメータを微分対象外にする
        for params in resnet.parameters():
            params.requires_grad = False
        # 最終層以外はそのまま利用
        self.model = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        # 最終層
        self.fc = nn.Linear(resnet.fc.in_features, metric_dim)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



# for img_path in img_dirs_test:
#     print(resnet(transformer(image).unsqueeze(0)).shape)
# print(transformer(image).shape)
# print(transformer(image).unsqueeze(0).shape)
# output = resnet(transformer(image).unsqueeze(0))
# print(output.shape)