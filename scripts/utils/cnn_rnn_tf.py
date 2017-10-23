# -*- coding: utf-8 -*-
# @Time    : 2017/10/20 下午4:36
# @Author  : Qi MO
# @Contact  : putao0124@gmail.com
# @Site  : http://blog.csdn.net/coraline_m
# @File    : cnn_rnn_tf.py
# @Software: PyCharm

import tensorflow as tf
from keras.layers import Embedding,Conv1D,Activation,BatchNormalization,MaxPool1D,Dropout
from keras.layers import Bidirectional,GRU,Flatten,Dense,ActivityRegularization
from keras.objectives import categorical_crossentropy,binary_crossentropy
from keras.regularizers import l2
from data_helper import load_data_cv
import yaml

class CNNRNN(object):
    def __init__(self,params):
        # 用户问题二维表示
        self.user_q = tf.placeholder(tf.float32,[None,params['user_q_dim']],name='user_q')
        # labels
        self.labels = tf.placeholder(tf.float32,[None,params['label']['dim']],name='labels')

        # 1.embedding layers
        embedding = Embedding(
            output_dim=params['embed_dim'],
            input_dim=params['vocab_size'],
            input_length=params['user_q_dim'],
            name='embedding',
            mask_zero=False)(self.user_q) # mask_zero会把ID为0的踢出，对rnn有用[不padding]
        embedding = BatchNormalization(momentum=0.9)(embedding)

        # 2.convolution for user_q
        if 'Conv1D' in params:
            conv_layer_num = len(params['Conv1D'])
            for i in range(1,conv_layer_num+1):
                H_input = embedding if i==1 else H
                conv = Conv1D(
                    filters=params['Conv1D']['layer%s'%i]['filters'],
                    kernel_size=params['Conv1D']['layer%s'%i]['filter_size'],
                    padding=params['Conv1D']['layer%s'%i]['padding_mode'],
                    # activation='relu,
                    strides=1,
                    bias_regularizer=l2(0.01))(H_input)
                # batch_norm
                conv_batch_norm = Activation('relu')(BatchNormalization(momentum=0.9)(conv))
                H = MaxPool1D(
                    pool_size=params['Conv1D']['layer%s'%i]['pooling_size'])(conv_batch_norm)
                # drop_out
                if 'dropout' in params['Conv1D']['layer%s'%i]:
                    H = Dropout(params['Conv1D']['layer%s'%i]['dropout'])(H)
        else:
            H = embedding

        # 3.Bi-LSTM
        if 'RNN' in params:
            rnn_cell = Bidirectional(
                GRU(units=params['RNN']['cell'],
                    dropout=params['RNN']['dropout'],
                    recurrent_dropout=params['RNN']['recurrent_dropout']))(H)
            H2 = rnn_cell
        else:
            H2 = Flatten()(H)

        # 4.predict probs for labels
        self.probs = Dense(
            units=params['label']['dim'],
            # activation='softmax',
            name = 'label_loss',
            bias_regularizer=l2(0.01))(H2)
        # batch_norm
        if 'batch_norm' in params['label']:
            self.probs = BatchNormalization(**params['label']['batch_norm'])(self.probs)
        self.probs = Activation(params['label']['loss_activate'])(self.probs)

        if 'activity_reg' in params['label']:
            self.probs = ActivityRegularization(
                name='label_activity_reg',
                **params['label']['activity_reg'])(self.probs)

        # 5.calculate_loss
        self.preds = tf.argmax(self.probs,axis=1,name='predictions')
        correct_predictions = tf.equal(
            tf.cast(self.preds,tf.int32),tf.cast(tf.argmax(self.labels,axis=1),tf.int32))
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_predictions,tf.float32),name='accuracy')
        self.loss = tf.reduce_mean(categorical_crossentropy(self.labels,self.probs),name='loss')

        # 6.set train_op
        if params['optimizer'] == 'adam':
            self.train_op = tf.train.AdamOptimizer(params['learn_rate']).minimize(self.loss)
        else:
            self.train_op = tf.train.RMSPropOptimizer(params['learn_rate']).minimize(self.loss)

if __name__ == '__main__':
    train_path = '../../docs/raw/Train_1000.csv'
    voc_dir = '../../docs/pro/'
    voc_mode = 1
    rev,voc_size = load_data_cv(train_path, voc_dir, voc_mode)

    params = yaml.load(open('params_tf.yaml','r',encoding='utf-8'))
    params['vocab_size'] = voc_size
    if params['model_type'] == 'char':
        params['user_q_dim'] = len(rev[0].char_user_q)
    else:
        params['user_q_dim'] = len(rev[0].word_user_q)
    cnn_rnn = CNNRNN(params)
