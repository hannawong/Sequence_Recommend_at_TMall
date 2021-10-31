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
- Try different models. Please refer to our Project Proposal.
- Now I only use RELU as activation function, we can try different activation functions such as PRelu and Dice. 
- In the baseline model, I only perform negative sampling when constructing the dataset. However, in real life scenarios where the data is a stream in online environment, it is infeasible to get all the data in advance. Therefore, a more practical way is to perform **in-batch** negative sampling. 
- Moreover, now I only use **random** negative sampling, which is the easiest way to perform negative sampling. However, this might make things too easy. We can also try other ways to do negative sampling, such as *hard negative sampling*, and negative sampling based on *streaming frequency estimation*. ([paper: Recsys 2019, Google](https://research.google/pubs/pub48840/))

