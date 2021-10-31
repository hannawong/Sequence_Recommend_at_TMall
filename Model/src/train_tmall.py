#coding:utf-8
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
import argparse
import pickle as pkl
import os

from data_iterator import DataIterator, generator_queue
from utils import calc_auc, prepare_data
from model import Model_DNN, Model_Vanilla_RNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--model_type', type=str, default='none', help='DNN | DIEN | MIMN | ..')


EMBEDDING_DIM = 16
HIDDEN_SIZE = 16 * 2 
best_auc = 0.0

    
def eval(sess, test_data, model, model_path, batch_size):
    loss_sum = 0.
    accuracy_sum = 0.
    nums = 0
    stored_arr = []
    test_data_pool, _stop, _ = generator_queue(test_data)
    while True:
        if  _stop.is_set() and test_data_pool.empty():
            break
        if not test_data_pool.empty():
            src,tgt = test_data_pool.get()
        else:
            continue
        user_id, item_id, cate_id, label, hist_item, hist_cate, hist_mask = prepare_data(src, tgt) 
        if len(user_id) < batch_size:
            continue
        nums += 1
        target = label
        prob, loss, acc, aux_loss = model.calculate(sess, [user_id, item_id, cate_id, hist_item, hist_cate,hist_mask, label])
        loss_sum += loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p ,t in zip(prob_1, target_1):
            stored_arr.append([p, t])
    
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        print("model save!!!!!!!")
        model.save(sess, model_path)
    return test_auc, loss_sum, accuracy_sum

def train(
        train_file = "../data/taobao_train.txt",
        test_file = "../data/taobao_test.txt",
        feature_file = "../data/taobao_feature.pkl",
        batch_size = 128,
        maxlen = 20,
        test_iter = 2000,
        model_type = 'DNN',
):

    best_model_path = "dnn_best_model/tmall_ckpt" + model_type
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
        train_data = DataIterator(train_file, batch_size, maxlen)
        test_data = DataIterator(test_file, batch_size, maxlen)   
        feature_num = pkl.load(open(feature_file,'rb'))  ##252260

        n_uid, n_mid = feature_num, feature_num
        BATCH_SIZE = batch_size
        SEQ_LEN = maxlen

        if model_type == 'DNN': 
            model = Model_DNN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == "Vanilla_RNN":
            model = Model_Vanilla_RNN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        else:
            print ("Invalid model_type : %s", model_type)
            return
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('training begin')

        iter = 0
        lr = 0.001
        for epoch in range(5):
            print("epoch  "+str(epoch))
            loss_sum = 0.0
            accuracy_sum = 0.
            train_data_pool,_stop,_ = generator_queue(train_data)
            while True:
                if  _stop.is_set() and train_data_pool.empty():
                    break
                if not train_data_pool.empty():
                    src,tgt = train_data_pool.get()
                else:
                    continue

                user_id, item_id, cate_id, label, hist_item, hist_cate, hist_mask = prepare_data(src, tgt)
                #print(user_id.shape,item_id.shape,cate_id.shape,label.shape,hist_item.shape,hist_cate.shape,hist_mask.shape)
                loss, acc, aux_loss = model.train(sess, [user_id, item_id, cate_id, hist_item, hist_cate, hist_mask, label, lr])
                loss_sum += loss
                accuracy_sum += acc
                iter += 1
                if (iter % test_iter) == 0:
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f' % \
                                          (iter, loss_sum / test_iter, accuracy_sum / test_iter))
                    print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ' % \
                                      eval(sess, test_data, model, best_model_path, batch_size))
                    loss_sum = 0.0
                    accuracy_sum = 0.0


if __name__ == '__main__':

    args = parser.parse_args()
    SEED = args.random_seed
    Model_Type = args.model_type ##DNN

    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train(model_type=Model_Type)
   