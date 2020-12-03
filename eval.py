import pickle
import random

from matplotlib import pyplot as plt
plt.switch_backend('agg')
import torch
from gensim.models import Word2Vec
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import OriginalResNet, TripletModel


# テスト用データセット
test_dataset_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/torch_dataset/test_dataset.pkl'
# word2vecの学習モデル
word2vec_path = "/mnt/LSTA5/data/tanaka/retrieval/text2image/word2vec.model"
# レシピコーパスで学習したWord2Vec
model = Word2Vec.load(word2vec_path)
# 学習済みモデル読み込み
head_model_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/model/'
#triplet_model = model.load_state_dict(torch.load(head_model_path + 'model_2048_50.pth', map_location=torch.device('cpu')))
triplet_model = TripletModel()
# GPU対応
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 損失関数
triplet_loss = nn.TripletMarginLoss()
# 学習済みのresnet
image_net = OriginalResNet()
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


# モデル評価
def eval_net(triplet_model,image_net, data_loader, dataset, loss, device="cpu"):
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
    fig.savefig("loss.png")


def main():
    # pickleでdatasetとdataloader読み込み 
    print('pickelでdataset読み込み中...')
    with open(test_dataset_path, 'rb') as f:
        test_dataset = pickle.load(f)
    print('完了')

    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    eval_net(triplet_model,image_net, test_loader, test_dataset, triplet_loss, device)

if __name__ == '__main__':
    main()


