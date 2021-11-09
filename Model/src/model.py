#coding:utf-8

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.keras.backend as K
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.keras.layers import (Dense,BatchNormalization,SimpleRNN)

from utils import VecAttGRUCell, feedforward, multihead_attention
from rnn import dynamic_rnn
from Transformer import transformer_model

class Model(object):
    def __init__(self, n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN, Flag="DNN", aux = False):
        
        self.model_flag = Flag
        self.aux = aux
        self.position_embedding_var = None

        with tf.name_scope('Inputs'):

            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')  # user_id
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph') # target_item_id
            self.cate_batch_ph = tf.placeholder(tf.int32, [None, ], name='cate_batch_ph') #target_cate_id
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph') ##item_history
            self.cate_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cate_his_batch_ph') # cate_history
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph') #history_mask
            self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph') #label:[0,1]/[1,0]
            self.lr = tf.placeholder(tf.float64, [])
            self.position_id = tf.placeholder(tf.int32,[None,None],name = "position_id") ## only useful in BERT and transformer

        # Embedding layer
        with tf.name_scope('Embedding_layer'):

            if self.model_flag == "BERT":
                self.position_embedding_var = tf.get_variable("position_embedding_var",[SEQ_LEN, HIDDEN_SIZE],trainable=True)

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM], trainable=True)  # Embedding table
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph) # embedding for target item id: (bs, 16)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph) # embedding for history item_id:(bs, maxlen, 16)

            self.cate_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.cate_batch_ph)# embedding for target cate id: (bs, 16)
            self.cate_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.cate_his_batch_ph)# embedding for history cate_id:(bs, maxlen, 16)          
            
        with tf.name_scope('init_operation'): 

            if self.model_flag == "BERT":
                self.position_embedding_placeholder = tf.placeholder(tf.float32,[SEQ_LEN, HIDDEN_SIZE], name="position_emb_ph")
                self.position_embedding_init = self.position_embedding_var.assign(self.position_embedding_placeholder)
            
            self.mid_embedding_placeholder = tf.placeholder(tf.float32,[n_mid, EMBEDDING_DIM], name="mid_emb_ph")
            self.mid_embedding_init = self.mid_embeddings_var.assign(self.mid_embedding_placeholder)
        

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cate_batch_embedded], axis=1) ## concat target item_id and cate: (bs,32)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded,self.cate_his_batch_embedded], axis=2) * tf.reshape(self.mask,(BATCH_SIZE, SEQ_LEN, 1)) ##(128,20,32), a sequence

        if self.aux:
            self.neg_iid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph') ##item_history
            self.neg_cate_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cate_his_batch_ph') # cate_history
            self.neg_item_his_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.neg_iid_his_batch_ph)
            self.neg_cate_his_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.neg_cate_his_batch_ph)
            self.neg_his_eb = tf.concat([self.neg_item_his_eb,self.neg_cate_his_eb], axis=2) * tf.reshape(self.mask,(BATCH_SIZE, SEQ_LEN, 1))   
        
        if self.model_flag == "BERT":
            self.position_id_embedding = tf.nn.embedding_lookup(self.position_embedding_var,self.position_id)
            self.item_his_eb = tf.add(self.item_his_eb,self.position_id_embedding)

    
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
            if self.aux:
                self.loss += self.aux_loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
    
    
    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask = None, stag = None):
        #mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]

        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask

        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.000001
        return y_hat

    def din_attention(self, query, facts, mask, sum = True):

        queries = tf.tile(query, [1, tf.shape(facts)[1]]) ##(?,640)
        queries = tf.reshape(queries, tf.shape(facts)) ##(128,20,32)
        din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)#(128,20,128)
        d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att') #(128,20,1)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]]) #(128,1,20)

        scores = d_layer_3_all

        if mask is not None:
            mask = tf.equal(mask, tf.ones_like(mask))
            key_masks = tf.expand_dims(mask, 1) # [Batchsize, 1, max_len]
            paddings = tf.ones_like(scores) * (-2 ** 32 + 1) # so that will be 0 after softmax
            scores = tf.where(key_masks, scores, paddings)  # if key_masks then scores else paddings

        scores = tf.nn.softmax(scores)  # [B, 1, T]
        if sum:
            output = tf.matmul(scores, facts)  # [B, 1, H]
            output = tf.squeeze(output)
        else:
            scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
            output = facts * tf.expand_dims(scores, -1)
            output = tf.reshape(output, tf.shape(facts))
            scores = tf.expand_dims(scores,-1)
        return output, scores


    def train(self, sess, inps):

        # feed_dict: user_id, item_id, cate_id, hist_item, hist_cate, hist_mask, label, lr
        if self.aux:
            loss, aux_loss, accuracy, _ = sess.run([self.loss, self.aux_loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cate_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cate_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.lr: inps[7],
                self.neg_iid_his_batch_ph : inps[8],
                self.neg_cate_his_batch_ph: inps[9],
                self.position_id: inps[10],
            })
        else:
            loss, accuracy, _ ,y_hat= sess.run([self.loss, self.accuracy, self.optimizer,self.y_hat], feed_dict={
                    self.uid_batch_ph: inps[0],
                    self.mid_batch_ph: inps[1],
                    self.cate_batch_ph: inps[2],
                    self.mid_his_batch_ph: inps[3],
                    self.cate_his_batch_ph: inps[4],
                    self.mask: inps[5],
                    self.target_ph: inps[6],
                    self.lr: inps[7],
                    self.position_id: inps[10],
                })
            aux_loss = 0
        return loss, accuracy, aux_loss            

    def calculate(self, sess, inps):
        if self.aux:
            probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cate_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cate_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.neg_iid_his_batch_ph: inps[7],
                self.neg_cate_his_batch_ph: inps[8],
                self.position_id: inps[9],
            })
        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                    self.uid_batch_ph: inps[0],
                    self.mid_batch_ph: inps[1],
                    self.cate_batch_ph: inps[2],
                    self.mid_his_batch_ph: inps[3],
                    self.cate_his_batch_ph: inps[4],
                    self.mask: inps[5],
                    self.target_ph: inps[6],
                    self.position_id: inps[9],
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

class Model_GRU4Rec(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_GRU4Rec, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="GRU")

        user_seq_hidden_state, user_seq_final_hidden_state = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb, 
                                dtype=tf.float32, scope='gru1')
        user_seq_hidden_state, user_seq_final_hidden_state = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=user_seq_hidden_state, 
                                dtype=tf.float32, scope='gru2')
        #GRU_output = GRU(HIDDEN_SIZE, return_sequences=True,go_backwards=True)(self.item_his_eb)
        GRU_output = user_seq_hidden_state * tf.reshape(self.mask,(BATCH_SIZE, SEQ_LEN, 1)) ## get masked
        GRU_mean_pooling = K.sum(GRU_output,axis = 1) / K.expand_dims(K.sum(self.mask, axis = 1))
        inp = tf.concat([self.item_eb,GRU_mean_pooling], 1)
        self.build_fcn_net(inp)

class MODEL_DIN(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(MODEL_DIN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="DIN")
        print(self.item_eb,self.item_his_eb,self.mask)
        attention_output,_ = self.din_attention(self.item_eb, self.item_his_eb, self.mask)
        inp = tf.concat([self.item_eb, attention_output], 1)
        self.build_fcn_net(inp)

class MODEL_DIEN(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256,aux = False):
        super(MODEL_DIEN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="DIEN",aux = aux)
        self.aux = aux
        rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         dtype=tf.float32,scope="gru1")

        if self.aux:
            aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                             self.neg_his_eb[:, 1:, :], self.mask[:, 1:], stag = "bigru_0")
            self.aux_loss = aux_loss_1

        attention_outputs, alphas = self.din_attention(self.item_eb, rnn_outputs, mask=self.mask, sum = False) #alpha: (128, 20)
        rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE),inputs = rnn_outputs, att_scores = alphas,
                                            dtype = tf.float32)
        mean_pooling = K.sum(rnn_outputs2,axis = 1) / K.expand_dims(K.sum(self.mask, axis = 1))
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb,axis = 1)
        inp = tf.concat([self.item_eb, final_state2, self.item_his_eb_sum, self.item_eb*self.item_his_eb_sum], 1)
        self.build_fcn_net(inp)


class MODEL_BERT(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(MODEL_BERT, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="BERT")
        
        Trm_outputs = transformer_model(self.item_his_eb,\
                                        hidden_size=HIDDEN_SIZE,\
                                        num_attention_heads=1,\
                                        num_hidden_layers=1,\
                                        intermediate_size=4*HIDDEN_SIZE,\
                                        hidden_dropout_prob=0.0,\
                                        attention_probs_dropout_prob=0.0
                                        )

        Bert_mean_pooling = K.sum(Trm_outputs,axis = 1) / K.expand_dims(K.sum(self.mask, axis = 1))
        inp = tf.concat([self.item_eb,Bert_mean_pooling], 1)
        self.build_fcn_net(inp)
        


        