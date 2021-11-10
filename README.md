# Sequential User Behavior Modeling for Click-Through Rate Prediction on TMALL dataset

## Prerequisites
- Python 3.x
- Tensorflow 2.x / 1.x

if you are using Tensorflow 1.x, just use `import tensorflow as tf`. if you are using Tensorflow 2.x, please use 
```py
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
``` 
instead.  

The model is run on NVIDIA GeForce 3090 with 8 GPUs, cuda version 11.4 . However, if you don't have GPU, you can also run the model on CPU. 

## Data prepare
Firstly, you need to put `UserBehavior.csv.zip` under directory /Model/data.
Then execute the following command in directory /Model/preprocess:
```sh
sh prepare_tmall.sh
```
 This would give you `taobao_feature.pkl`,`taobao_test.txt` and `taobao_train.txt`. 
It might take 20 minutes to preprocess full dataset. However, you can use `UserBehavior_sample.csv` instead, it has fewer samples and can be processed in seconds. 

## Run base model
In directory /src, run the following command:
```
python train_tmall.py --model_type DNN
```

## Ways to improve

### Try different models
- Simple Deep Neural Network(DNN) with mean pooling. Easy √
- Simple RNN. Easy √
- LSTM. (Already integrated into Keras Library) Easy
- GRU4Rec. (Already integrated into Keras Library.[paper: ICLR 2016](https://arxiv.org/pdf/1511.06939.pdf)) Easy √
- Transformer4Rec. (Use code from [Huggingface project](https://github.com/huggingface/transformers). [paper: Google, NIPS 2017](https://arxiv.org/pdf/1706.03762.pdf)) Medium 
- Caser. (code available in [github](https://github.com/graytowne/caser). [paper: WSDM 2018](https://arxiv.org/pdf/1809.07426.pdf)) Medium
- DIN. (code available in [github](https://github.com/zhougr1993/DeepInterestNetwork). [paper:Taobao, KDD 2018](https://arxiv.org/pdf/1706.06978.pdf)) Medium √
- DIEN. (code available in [github](https://github.com/mouna99/dien). [paper: Taobao, AAAI 2019](https://arxiv.org/pdf/1809.03672.pdf)) Hard √
- BERT4Rec. (code available in [github](https://github.com/FeiSun/BERT4Rec). [paper: Alibaba, CIKM 2019](https://arxiv.org/abs/1904.06690))
Very Hard


### Negative sampling trick
- In the baseline model, I only perform negative sampling when constructing the dataset. However, in real life scenarios where the data is a stream in online environment, it is infeasible to get all the data in advance. Therefore, a more practical way is to perform **in-batch** negative sampling. 

- Moreover, now I only use **random** negative sampling, which is the easiest way to perform negative sampling. However, this might make things too easy. We can also try other ways to do negative sampling, such as *hard negative sampling*, and negative sampling based on *streaming frequency estimation*. ([paper: Recsys 2019, Google](https://research.google/pubs/pub48840/))

- It is also possible to combine in-batch negative sampling with random negative sampling and balance those two method with a hyperparameter alpha. Please refer to: [paper: 2021, JD.com](https://arxiv.org/pdf/2006.02282.pdf)

### Try Different activation functions

Now I only use RELU as activation function, we can try different activation functions such as PRelu, Dice and Mish(?). 




