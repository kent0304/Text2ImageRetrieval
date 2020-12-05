import pickle
import numpy as np
import random

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from scipy import spatial

from model import OriginalResNet

# 1000のレシピ文用意
def prepare_recipes():
    # testデータセット読み込み
    with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    image_data = []
    random.seed(0)
    idx_list = random.sample(range(len(test_dataset)), k=1000)
    for idx in idx_list:
        image_data.append(test_dataset[idx][1])
    return image_data

# 画像間の類似度
def image_similarity(img1_vec, img2_vec):
    return 1 - spatial.distance.cosine(img1_vec, img2_vec)

def image2vec(image_path):
    image_net = OriginalResNet()
    # 画像を Tensor に変換
    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.RandomCrop(256),
    ])
    # 画像のベクトル化
    image = transformer(Image.open(image_path).convert('RGB')).unsqueeze(0)
    image = image_net(image).numpy()
    return image

def measurement(target, image_data):
    # 辞書の値からキー抽出
    def get_key_from_value(d, val):
        keys = [k for k, v in d.items() if v == val]
        if keys:
            return keys[0]
        return None

    # 類似度大きい順から10個管理
    similar_dict = {}
    for data in tqdm(image_data, total=len(image_data)):
        if data == target:
            continue
        else:
            similarity = image_similarity(image2vec(target), image2vec(data))
            if len(similar_dict) < 10:
                similar_dict[data] = similarity
            else:
                if min(similar_dict.values()) < similarity:
                    key = get_key_from_value(similar_dict, min(similar_dict.values()))
                    del similar_dict[key]
                    similar_dict[data] = similarity
    return sorted(similar_dict.items(), key=lambda x:x[1])


def main():
    image_data = prepare_recipes()
   
    similar_dict = measurement(image_data[0], image_data)
    similar_dict2 = measurement(image_data[1], image_data)

    print(f'{image_data[0]}に近い画像top10')
    for k, v in similar_dict.items():
        print(k, v)
    print('')
    print(f'{image_data[1]}に近い画像top10')
    for k, v in similar_dict2.items():
        print(k, v)


if __name__ == '__main__':
    main()

