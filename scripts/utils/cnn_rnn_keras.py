# -*- coding: utf-8 -*-
# @Time    : 2017/10/22 下午4:34
# @Author  : Qi MO
# @Contact  : putao0124@gmail.com
# @Site  : http://blog.csdn.net/coraline_m
# @File    : cnn_rnn_keras.py
# @Software: PyCharm

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate

class CNNRNN(object):
    def __init__(self,params):
        """

        :param params:
        """

        user_q_dim = params['user_q_dim']
        embedding_dim = params['embed_dim']
        vocab_size = params['vocab_size']
        model_input = Input(shape=(user_q_dim,))

        # 1. Embedding
        embedding = Embedding(vocab_size, embedding_dim, input_length=user_q_dim, name="embedding")(model_input)
        embedding = Dropout(params['drop_out_1'])(embedding)

        # 2. Convolution and Pooling
        filter_sizes =  (2, 3, 4)
        conv_blocks = []
        for sz1 in filter_sizes:
            conv_1 = Convolution1D(filters=params['filters'],
                                 kernel_size=sz1,
                                 padding=params['padding_mode'],
                                 activation=params['activation'],
                                 strides=params['strides'])(embedding)
            conv_1 = MaxPooling1D(pool_size=params['pool_size_1'])(conv_1)

            for sz2 in filter_sizes:
                conv_2 = Convolution1D(filters=params['filters'],
                                     kernel_size=sz2,
                                     padding=params['padding_mode'],
                                     activation=params['activation'],
                                     strides=params['strides'])(conv_1)
                conv_2 = MaxPooling1D(pool_size=params['pool_size_1'])(conv_2)
                conv_2 = Flatten()(conv_2)
                conv_blocks.append(conv_2)

        H = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        H = Dropout(params['drop_out_2'])(H)

        # 3. oupt_out层
        model_output = Dense(params['label_dim'], activation="softmax")(H)

        # 4. 构建model
        self.model = Model(model_input, model_output)
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
