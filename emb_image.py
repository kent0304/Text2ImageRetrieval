import torch
from torch import nn, optim
import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models

# 入力画像のpath
path = ''

# 事前学習済みのresnet50をダウンロード
resnet = models.resnet50(pretrained=True)

# Datasetを作成
imgs_dataset = ImageFolder(
    "data/images/train",
    transform = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)


# DataLoaderを作成
train_loader = DataLoader(train_imgs, batch_size=1, shuffle=True)
test_loader = DataLoader(test_imgs, batch_size=32, shuffle=False)