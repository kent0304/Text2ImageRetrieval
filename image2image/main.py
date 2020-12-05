import pickle
import numpy as np
import random

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
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


def image_similarity(v1, v2):
    return  np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def image2vec(image_path):
    # ブログ参考
    image_net = models.resnet50(pretrained=True)
    image_net.fc = nn.Identity()

    # 画像を Tensor に変換
    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 画像のベクトル化
    image_net.eval()
    image = transformer(Image.open(image_path).convert('RGB')).unsqueeze(0)
    image = image_net(image).detach().numpy().copy()
    return image.flatten()

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

    print(image_similarity(image2vec('/mnt/LSTA5/data/sakoda/image_step_dataset/3ae13f68982975988da11927410c4465b3104bf2/3ae13f68982975988da11927410c4465b3104bf2_9.jpg'),  image2vec('/mnt/LSTA5/data/sakoda/image_step_dataset/45f31a2c32e015d814365ae66a3a77d5a38c8145/45f31a2c32e015d814365ae66a3a77d5a38c8145_5.jpg')))
   
    similar_dict = measurement(image_data[100], image_data)
    print(f'{image_data[100]}に近い画像top10')
    for _ in similar_dict:
        print(_)

    print('')
    dog = '/home/tanaka/projects/retrieval/dog.jpg'
    print(image_similarity(image2vec(image_data[100]), image2vec(dog)))


if __name__ == '__main__':
    main()

