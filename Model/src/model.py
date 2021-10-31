#coding:utf-8

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (Dense,BatchNormalization,SimpleRNN)

class Model(object):
    def __init__(self, n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN, Flag="DNN"):
        
        self.model_flag = Flag

        with tf.name_scope('Inputs'):

            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')  # user_id
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph') # target_item_id
            self.cate_batch_ph = tf.placeholder(tf.int32, [None, ], name='cate_batch_ph') #target_cate_id
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph') ##item_history
            self.cate_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cate_his_batch_ph') # cate_history
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph') #history_mask
            self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph') #label:[0,1]/[1,0]
            self.lr = tf.placeholder(tf.float64, [])

        # Embedding layer
        with tf.name_scope('Embedding_layer'):

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM], trainable=True)  # Embedding table
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph) # embedding for target item id: (bs, 16)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph) # embedding for history item_id:(bs, maxlen, 16)

            self.cate_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.cate_batch_ph)# embedding for target cate id: (bs, 16)
            self.cate_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.cate_his_batch_ph)# embedding for history cate_id:(bs, maxlen, 16)          
            
        with tf.name_scope('init_operation'):    
            self.mid_embedding_placeholder = tf.placeholder(tf.float32,[n_mid, EMBEDDING_DIM], name="mid_emb_ph")
            self.mid_embedding_init = self.mid_embeddings_var.assign(self.mid_embedding_placeholder)
        
        self.item_eb = tf.concat([self.mid_batch_embedded, self.cate_batch_embedded], axis=1) ## concat target item_id and cate: (bs,32)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded,self.cate_his_batch_embedded], axis=2) * tf.reshape(self.mask,(BATCH_SIZE, SEQ_LEN, 1)) ##(128,20,32), a sequence

    def build_fcn_net(self, inp):
        ## TODO: activation function Prelu/Dice
        bn1 = BatchNormalization()(inp)
        dnn1 = Dense(200, activation="relu", name='f1')(bn1)
        dnn2 = Dense(80, activation="relu", name='f2')(dnn1)
        dnn3 = Dense(2, activation=None, name='f3')(dnn2)
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.reg_loss =  tf.losses.get_regularization_loss()
            self.loss = tf.add(ctr_loss,self.reg_loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            
    def train(self, sess, inps):

        # feed_dict: user_id, item_id, cate_id, hist_item, hist_cate, hist_mask, label, lr

        loss, accuracy, _ ,y_hat= sess.run([self.loss, self.accuracy, self.optimizer,self.y_hat], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cate_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cate_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.lr: inps[7]
            })
        aux_loss = 0
        return loss, accuracy, aux_loss            

    def calculate(self, sess, inps):
        probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cate_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cate_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6]
        })
        aux_loss = 0
        return probs, loss, accuracy, aux_loss

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

class Model_DNN(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_DNN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="DNN")

        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1) # sum-pooling, for DNN
        inp = tf.concat([self.item_eb, self.item_his_eb_sum], 1)
        self.build_fcn_net(inp)
        
class Model_Vanilla_RNN(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_Vanilla_RNN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="RNN")

        RNN_output = SimpleRNN(HIDDEN_SIZE,return_sequences=True)(self.item_his_eb)
        RNN_output = RNN_output * tf.reshape(self.mask,(BATCH_SIZE, SEQ_LEN, 1)) ## get masked
        RNN_mean_pooling = K.sum(RNN_output,axis = 1) / K.expand_dims(K.sum(self.mask, axis = 1))
        inp = tf.concat([self.item_eb, RNN_mean_pooling], 1)
        self.build_fcn_net(inp)

        