# テキスト情報はtrain, valid, testにわけ, data/textに格納
# 画像はgeneratorでtrain, vali, testごとにパスを用意

import os
import csv
import glob
from multiprocessing import Pool
from tqdm import tqdm 


# ファイルの読み込み
def read_file(file):
    with open(file) as f:
        step = [line.strip() for line in f.readlines()]
        step = [line.split('\t')[1] if '\t' in line else line for line in step]
    return step

# フォルダ内の画像をジェネレータで出力
def listup_imgs(path):
    yield [os.path.abspath(p) for p in glob.glob(path)]

# データセットのパス
cookpad_path = '/mnt/LSTA5/data/common/recipe/cookpad_image_dataset/'
recipe_train = 'recipe_ids/train.txt'
recipe_valid = 'recipe_ids/val.txt'
recipe_test = 'recipe_ids/test.txt'
file_relation = 'cookpad_nii/main/file_relation_list.txt'
recipe_info = 'cookpad_nii/main/'
img_path = 'images/steps/'

# テキストデータ用意 ----------------------------------------------------------------------------------------------------
# 訓練用recipe_ids（ハッシュ列） 読み込み
with open(os.path.join(cookpad_path, recipe_train)) as f:
    recipe_train = [line.strip() for line in f.readlines()]
# 検証用recipe_ids（ハッシュ列） 読み込み
with open(os.path.join(cookpad_path, recipe_valid)) as f:
    recipe_valid = [line.strip() for line in f.readlines()]
# 試験用recipe_ids（ハッシュ列） 読み込み
with open(os.path.join(cookpad_path, recipe_test)) as f:
    recipe_test = [line.strip() for line in f.readlines()]

# recipe_ids（ハッシュ列） とそれがどのディレクトリに保存されてるか記してるファイル読み込み
with open(os.path.join(cookpad_path, file_relation)) as f:
    file_relation_list = [line.strip().split('\t') for line in f.readlines()]

# recipeid と 保存されているディレクトリ のペアの dictionary
relation_dict = {k: v for (k,v) in file_relation_list}

# レシピのテキストデータを二次元配列で格納
step_text_train = []
step_text_valid = []
step_text_test = []

# 訓練用レシピのテキストファイルのパス管理
recipe_dirs_train = []
for recipe in recipe_train:
    recipe_dir = relation_dict[recipe]
    # 3桁いないのディレクトリ名は前にを足してパス変更
    if len(recipe_dir) < 4:
        num = 4 - len(recipe_dir)
        recipe_dir = cookpad_path + recipe_info + '0' * num + recipe_dir + '/' + recipe + '/' + 'step_memos.txt'
        recipe_dirs_train.append(recipe_dir)
    else:
        continue
# 検証用レシピのテキストファイルのパス管理
recipe_dirs_valid = []
for recipe in recipe_valid:
    recipe_dir = relation_dict[recipe]
    # 3桁いないのディレクトリ名は前にを足してパス変更
    if len(recipe_dir) < 4:
        num = 4 - len(recipe_dir)
        recipe_dir = cookpad_path + recipe_info + '0' * num + recipe_dir + '/' + recipe + '/' + 'step_memos.txt'
        recipe_dirs_valid.append(recipe_dir)
    else:
        continue
# 試験用レシピのテキストファイルのパス管理
recipe_dirs_test = []
for recipe in recipe_test:
    recipe_dir = relation_dict[recipe]
    # 3桁いないのディレクトリ名は前にを足してパス変更
    if len(recipe_dir) < 4:
        num = 4 - len(recipe_dir)
        recipe_dir = cookpad_path + recipe_info + '0' * num + recipe_dir + '/' + recipe + '/' + 'step_memos.txt'
        recipe_dirs_test.append(recipe_dir)
    else:
        continue

# # マルチプロセッシング
# with Pool() as p:
#     step_text_train = p.map(read_file, recipe_dirs_train)
#     step_text_valid = p.map(read_file, recipe_dirs_valid)
#     step_text_test = p.map(read_file, recipe_dirs_test)



# # 学習用レシピテキストファイル書き込み
# with open('data/text/train.txt', mode='w') as f:
#     for steps in step_text_train:
#         f.write('\t'.join(steps))
#         f.write('\n')
# # 検証用レシピテキストファイル書き込み
# with open('data/text/valid.txt','w') as f:
#     for steps in step_text_valid:
#         f.write('\t'.join(steps))
#         f.write('\n')
# # # 試験用レシピテキストファイル書き込み
# with open('data/text/test.txt','w') as f:
#     for steps in step_text_test:
#         f.write('\t'.join(steps))
#         f.write('\n')



# ここから画像データ用意------------------------------------------------------------------------------------------------
# 学習用のレシピの画像のパスを管理
img_dirs_train = []
for recipe in recipe_train:
    head = recipe[0]
    path = cookpad_path + img_path + head + '/' + recipe + '*'
    img_dir_train = listup_imgs(path)
    img_dirs_train.append(img_dir_train)
# 検証用のレシピの画像のパスを管理
img_dirs_valid = []
for recipe in recipe_valid:
    head = recipe[0]
    path = cookpad_path + img_path + head + '/' + recipe + '*'
    img_dir_valid = listup_imgs(path)
    img_dirs_valid.append(img_dir_valid)
# 試験用のレシピの画像のパスを管理
img_dirs_test = []
for recipe in recipe_test:
    head = recipe[0]
    path = cookpad_path + img_path + head + '/' + recipe + '*'
    img_dir_test = listup_imgs(path)
    img_dirs_test.append(img_dir_test)

# img_dirs_train, img_dirs_valid, img_dirs_test はそれぞれ二次元配列
# 各要素は generator なので注意
