# Image Retrieval with text
I implemented text2image on retrieval-base by using triplet margin loss function with PyTorch. Image retrieval is a kind of tasks to find images related to a given query text. A common strategy to learn those similarities is to learn embeddings of images and queries in the same vectorial space (often called embedding space). 
In my project, that would be learning embeddings of cooking images and vectors encoding recipe procedures in the same space. However I cannot upload this dataset. So you need to replace some dataset inside of my code.

## Getting stated
First you have to create original Word2Vec model for your data model in word2vec.py

```bash
python word2vec.py
```

Second you prepare your original dataset. Defined PyTorch Dataset in prepare_data.py.

``` bash
python prepare_data.py
```

You could just saved your dataset by pickle this time.

Finally you could train triplet model defined in model.py.

```bash
python train.py 
```

You can check the model, losses list and line graph at the following path 

```python
# 学習済みモデル保存パス先頭部分
head_model_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/model/'
# 学習時のloss値保存先
train_losses_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/losses/train_losses.2048.pkl'
valid_losses_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/losses/valid_losses.2048.pkl'
# loss画像保存先
losspng_path = '/mnt/LSTA5/data/tanaka/retrieval/text2image/loss_png'
```

# Result
## Baseline
- Recall@K
- MedR: 97

## with Stopwords
### with word2vec100
- Recall@K:  [0.018, 0.065, 0.108, 0.361, 0.521]
- MedR: 92
