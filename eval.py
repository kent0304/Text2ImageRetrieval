import pickle
import random
import statistics

import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import torch
from gensim.models import Word2Vec
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model import TripletModel

# GPU対応
device = torch.device('cuda:1')
# テスト用データセット
test_dataset_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/test_dataset.pkl'
# # word2vecの学習モデル
# word2vec_path = "/mnt/LSTA5/data/tanaka/retrieval/text2image/word2vec.model"
# # レシピコーパスで学習したWord2Vec
# model = Word2Vec.load(word2vec_path)
# 学習済みモデル読み込み
head_model_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/model/'
triplet_model = TripletModel()
triplet_model.load_state_dict(torch.load(head_model_path + 'stopwords300_model_epoch400.pth', map_location=device))
# 損失関数
triplet_loss = nn.TripletMarginLoss()

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

class MyDataset(Dataset):
    def __init__(self, sentence_vec, image_vec):
        if len(sentence_vec) != len(image_vec):
            print('一致してない')
            exit(0)
        self.data_num = len(sentence_vec)
        self.sentence_vec = sentence_vec
        self.image_vec = image_vec
        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        sentence_vec = self.sentence_vec[idx]
        image_vec = self.image_vec[idx]
        return sentence_vec, image_vec

# コサイン類似度
def cos_sim(a, b) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
# Recall@Kの算出
def recall_text2image(triplet_model, dataset, k_list, device=device) -> list:
    # 辞書の値からキー抽出
    def get_key_from_value(d, val) -> str:
        keys = [k for k, v in d.items() if v == val]
        if keys:
            return keys[0]
        return None

    triplet_model.eval()
    triplet_model = triplet_model.to(device)
    data_num = len(dataset)
    recall_list = []
    for k in k_list:  
        
        score = 0
        for i in tqdm(range(data_num), total=data_num):
            sim_dict = {}
            for j in range(data_num):
                text = dataset[i][0].to(device)
                image = dataset[j][1].to(device)
                text_vec, image_vec, _, _ = triplet_model(text.float(), image.float(), text.float(), image.float())
                similarity = cos_sim(text_vec.cpu().detach().numpy(), image_vec.cpu().detach().numpy())
                if len(sim_dict) < k:
                    sim_dict[str(j)] = similarity
                else:
                    if min(sim_dict.values()) < similarity:
                        key = get_key_from_value(sim_dict, min(sim_dict.values()))
                        del sim_dict[key]
                        sim_dict[str(j)] = similarity
            # text i に対して最も近い image j の辞書 にiが含まれるか
            if str(i) in sim_dict.keys():
                score += 1
        # kでのrecall算出
        recall = score / data_num 
        recall_list.append(recall)
    return recall_list

def recall_image2text(triplet_model, dataset, k_list, device=device) -> list:
    # 辞書の値からキー抽出
    def get_key_from_value(d, val) -> str:
        keys = [k for k, v in d.items() if v == val]
        if keys:
            return keys[0]
        return None

    triplet_model.eval()
    triplet_model = triplet_model.to(device)
    data_num = len(dataset)
    recall_list = []
    for k in k_list:  
        
        score = 0
        for i in tqdm(range(data_num), total=data_num):
            sim_dict = {}
            for j in range(data_num):
                text = dataset[j][0].to(device)
                image = dataset[i][1].to(device)
                text_vec, image_vec, _, _ = triplet_model(text.float(), image.float(), text.float(), image.float())
                similarity = cos_sim(text_vec.cpu().detach().numpy(), image_vec.cpu().detach().numpy())
                if len(sim_dict) < k:
                    sim_dict[str(i)] = similarity
                else:
                    if min(sim_dict.values()) < similarity:
                        key = get_key_from_value(sim_dict, min(sim_dict.values()))
                        del sim_dict[key]
                        sim_dict[str(i)] = similarity
            # text i に対して最も近い image j の辞書 にiが含まれるか
            if str(i) in sim_dict.keys():
                score += 1
        # kでのrecall算出
        recall = score / data_num 
        recall_list.append(recall)
    return recall_list

# MedR

def medr_text2image(triplet_model, dataset, device=device) -> list:
    triplet_model.eval()
    triplet_model = triplet_model.to(device)
    data_num = len(dataset)
 
    med_list = []
    for i in tqdm(range(data_num), total=data_num):
        sim_dict = {}
        for j in range(data_num):
            text = dataset[i][0].to(device)
            image = dataset[j][1].to(device)
            text_vec, image_vec, _, _ = triplet_model(text.float(), image.float(), text.float(), image.float())
            similarity = cos_sim(text_vec.cpu().detach().numpy(), image_vec.cpu().detach().numpy())
            sim_dict[str(j)] = similarity
        sim_list = sorted(sim_dict.items(), key=lambda x:x[1], reverse=True)
        idx = sim_list.index([tup for tup in sim_list if tup[0] == str(i)][0])+1
        med_list.append(idx)

    return statistics.median(med_list)


            

        


def main():
    print('pickelでdataset読み込み中...')
    data_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/'
    # # sentence_vec 読み込み
    # with open(data_path + 'test_sentence_vec.pkl', 'rb') as f:
    #     test_sentence_vec = pickle.load(f)
    with open(data_path + 'word2vec300/test_sentence_vec.pkl', 'rb') as f:
        test_sentence_vec = pickle.load(f)
    # image_vec 読み込み
    with open(data_path + 'test_image_vec.pkl', 'rb') as f:
        test_image_vec = pickle.load(f)
    # PyTorch の Dataset に格納
    test_dataset = MyDataset(test_sentence_vec, test_image_vec)
    test_dataset = random.sample(list(test_dataset), 1000)

    # Recall@K
    k_list = [1, 5, 10, 50, 100]
    recall_text2image_list = recall_text2image(triplet_model=triplet_model, dataset=test_dataset, k_list=k_list)
    print('Recall@K(text2image):', recall_text2image_list)
    # recall_image2text_list = recall_text2image(triplet_model=triplet_model, dataset=test_dataset, k_list=k_list)
    # print('Recall@K(image2text):', recall_image2text_list)



    # medr
    medr = medr_text2image(triplet_model=triplet_model, dataset=test_dataset)
    print('MedR:',int(medr))
        
    print('word2vecを100次元にしてstopwordsも導入したときの結果です。')
   




    # MedR

if __name__ == '__main__':
    main()


