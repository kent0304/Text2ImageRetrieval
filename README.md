## Image Retrieval with text
I implemented text2image on retrieval-base by using triplet margin loss function with PyTorch. Image retrieval is a kind of tasks to find images related to a given query text. 
### Data
I used cooking data(recipe procedures text and images), however I cannot upload this dataset. So you need to replace inside of my code.

### Getting stated
First you have to create your original Word2Vec model in word2vec.py

``` python word2vec.py```

Second you prepare your original dataset. Defined PyTorch Dataset in prepare_data.py.

``` python prepare_data.py```

You could just saved your dataset by pickle this time.

Finally you could train triplet model defined in model.py.

``` python train.py ```


