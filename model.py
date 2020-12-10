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


class TripletModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_fc1 = nn.Linear(2048, 1536)
        self.image_fc2 = nn.Linear(1536, 1024)
        self.image_fc3 = nn.Linear(1024, 512)

        self.text_fc1 = nn.Linear(300, 372)
        self.text_fc2 = nn.Linear(372, 446)
        self.text_fc3 = nn.Linear(446, 512)

    def forward(self, pos_text, pos_image, neg_text, neg_image):
        pos_text = F.relu(self.text_fc1(pos_text))
        pos_text = F.relu(self.text_fc2(pos_text))
        pos_text_vec = self.text_fc3(pos_text)

        pos_image = F.relu(self.image_fc1(pos_image))
        pos_image = F.relu(self.image_fc2(pos_image))
        pos_image_vec = self.image_fc3(pos_image)

        neg_text = F.relu(self.text_fc1(neg_text))
        neg_text = F.relu(self.text_fc2(neg_text))
        neg_text_vec = self.text_fc3(neg_text)

        neg_image = F.relu(self.image_fc1(neg_image))
        neg_image = F.relu(self.image_fc2(neg_image))
        neg_image_vec = self.image_fc3(neg_image)

        return pos_text_vec, pos_image_vec, neg_text_vec, neg_image_vec
