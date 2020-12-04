from tqdm import tqdm
import pickle
import shutil

# 画像コピー先ディレクトリ
train_image_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/image_path/train'
valid_image_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/image_path/valid'
test_image_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/image_path/test'

# 画像をそれぞれ指定のディレクトリにコピーする関数
def mydataset(dataset, dir):
    data_num = len(dataset)
    for i, (text, image_path) in tqdm(enumerate(dataset), total=data_num):
        if text=='' or image_path=='':
            continue
        # 画像コピー
        shutil.copy(image_path, dir)
        

# pickleで読み込み
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/train_dataset.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/valid_dataset.pkl', 'rb') as f:
    valid_dataset = pickle.load(f)
with open('/mnt/LSTA5/data/tanaka/retrieval/text2image/dataset/test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)


print('train valid testそれぞれの画像をそれぞれのパスにコピー中...')
train_dataset = mydataset(train_dataset, train_image_path)
valid_dataset = mydataset(valid_dataset, valid_image_path)
test_dataset = mydataset(test_dataset, test_image_path)
