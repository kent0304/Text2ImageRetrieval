import torch
from torch import nn
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms, models

# GPU対応
device = torch.device('cuda:0')
# device = torch.device('cpu')

# pickleで読み込み
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/train_dataset.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/valid_dataset.pkl', 'rb') as f:
    valid_dataset = pickle.load(f)
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)

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

def mydataset(dataset):
    # resnet呼び出し
    image_net = models.resnet50(pretrained=True)
    image_net.fc = nn.Identity()
    image_net.eval()
    image_net = image_net.to(device)
    data_num = len(dataset)
    image_vec = torch.zeros((data_num, 2048))
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            image_paths = [dataset[j][1] for j in range(i, len(dataset))[:batch_size]]
            images = image2vec(image_net, image_paths)
            image_vec[i:i + batch_size] = images

            # if i >= 10*batch_size:
            #     exit(0)
        
    return image_vec


valid_image_vec = mydataset(valid_dataset)
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/valid_image_vec.pkl', 'wb') as f:
    pickle.dump(valid_image_vec, f) 
print('valid終了')

test_image_vec = mydataset(test_dataset)
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/test_image_vec.pkl', 'wb') as f:
    pickle.dump(test_image_vec, f) 
print('test終了')

train_image_vec = mydataset(train_dataset)
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/train_image_vec.pkl', 'wb') as f:
    pickle.dump(train_image_vec, f, protocol=4) 
print('train終了')

