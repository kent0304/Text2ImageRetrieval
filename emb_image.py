import torch
from torch import nn, optim
import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models
from PIL import Image
from prepare_data import img_dirs_train, img_dirs_valid, img_dirs_test

# 入力画像のpath
path = '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/images/steps/0/0fffdec2d05716551f0531714536cba6c235a7d0_7.jpg'

image = Image.open(path)
# Tensor に変換
transformer = transforms.Compose([
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 事前学習済みのresnet50をダウンロード
resnet = models.resnet50(pretrained=True)
# # 最終層カット
# resnet = list(resnet.children())[:-1]
# resnet = nn.Sequential(*resnet)
# 最後の線形層を付け替える
fc_input_dim = resnet.fc.in_features
resnet.fc = nn.Linear(fc_input_dim, 300)
# 全てのパラメータを微分対象外にする
for p in resnet.parameters():
    p.requires_grad = False

for img_path in img_dirs_test:
    print(resnet(transformer(image).unsqueeze(0)).shape)
# print(transformer(image).shape)
# print(transformer(image).unsqueeze(0).shape)
# output = resnet(transformer(image).unsqueeze(0))
# print(output.shape)