# -*- coding: utf-8 -*-
# @Time    : 2017/10/20 下午6:42
# @Author  : Qi MO
# @Contact  : putao0124@gmail.com
# @Site  : http://blog.csdn.net/coraline_m
# @File    : train_tf.py
# @Software: PyCharm

import yaml
import json
import time
import os
import tensorflow as tf
import numpy as np
from eval_tf import do_eval
from utils.cnn_rnn_tf import CNNRNN
from utils.data_helper import load_data_cv
from keras import backend as K

K.set_learning_phase(1) # https://keras.io/backend/ 1表示训练，0表示测试

def train(params):
    """
    train and eval
    :param params:  util中params参数
    :return:
    """

    datas,voc_size = load_data_cv(
        file_path='../docs/raw/Train.csv',
        voc_dir = '../docs/pro/',
        voc_mode=1, # 如果使用词模型，这个地方改为2, params.yaml文件中model_type改为word
        cv=10)

    params['vocab_size'] = voc_size
    if params['model_type'] == 'char':
        params['user_q_dim'] = len(datas[0].char_user_q)
    else:
        params['user_q_dim'] = len(datas[0].word_user_q)
    # indent=4 格式化输出，check params
    print(json.dumps(params,indent=4))

    dev_datas = list(filter(lambda data: data.cv==1, datas))
    train_datas = list(filter(lambda data: data.cv!=1, datas))

    print('len train datas: {}'.format(len(train_datas)))
    print('len dev datas: {}'.format(len(dev_datas)))

    print('user_q of train[0]: {}'.format(train_datas[0].char_user_q))
    print('labels of train[0]: {}'.format(train_datas[0].labels))

    number_of_training_data = len(train_datas)
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S",time.localtime())
    # log for tensorboard visualization
    # log_dev_dir = '../docs/tensorflow/dev/%s' % timestamp
    # log_train_dir = '../docs/tensorflow/train/%s' % timestamp
    # os.mkdir(log_dev_dir),os.mkdir(log_train_dir)

    batch_size = params['batch_size']

    # 设置gpu限制 参考 http://hpzhao.com/2016/10/29/TensorFlow%E4%B8%AD%E8%AE%BE%E7%BD%AEGPU/
    config = tf.ConfigProto(allow_soft_placement=True)  # 自动选择一个存在并支持的gpu
    config.gpu_options.per_process_gpu_memory_fraction = 0.6

    # add model saver, default save latest 4 model checkpoints
    model_dir = params['model_dir'] + time.strftime("%Y-%m-%d_%H:%M:%S",time.localtime())
    os.mkdir(model_dir)
    model_path = '{}/{}'.format(model_dir,params['model_name'])
    print(model_path)

    with tf.Session(config=config) as sess, tf.device('/gpu:0'):
        cnn_rnn = CNNRNN(params)

        saver = tf.train.Saver(max_to_keep=4)
        # dev_writer = tf.summary.FileWriter(log_dev_dir)
        # train_writer = tf.summary.FileWriter(log_train_dir)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        step = -1
        best_acc,best_step = 0.,0
        for epoch in range(params['epoch']):
            # shuffle in each epoch
            train_datas = np.random.permutation(train_datas)

            for start in range(0,number_of_training_data,batch_size):
                end = start+batch_size
                step += 1
                if params['model_type'] == 'char': # 字模型
                    user_q = [QM.char_user_q for QM in train_datas[start:end]]
                else: # 词模型
                    user_q = [QM.word_user_q for QM in train_datas[start:end]]
                labels = [QM.labels for QM in train_datas[start:end]]

                trn_loss,trn_probs,trn_acc = sess.run(
                    [cnn_rnn.loss,cnn_rnn.probs,cnn_rnn.accuracy],
                    feed_dict={
                        cnn_rnn.user_q: user_q,
                        cnn_rnn.labels: labels
                    })

                #每 log_train_batch 记录训练集上的loss和acc到tensorboard
                if step % params['log_train_batch'] == 0:
                    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
                    str_loss = '{}: epoch:{},step:{},train_loss:{},train_acc:{}'.format(
                        timestamp, epoch, step, trn_loss, trn_acc)
                    print(str_loss)
                    # train_writer.add_summary(
                    #     tf.Summary(value=[
                    #         tf.Summary.Value(tag="loss",simple_value=trn_loss),
                    #         tf.Summary.Value(tag="accuracy", simple_value=trn_acc),
                    #     ]),step)
                #每 eval_test_batch 记录发展集（验证集）上的loss和acc到tensorboard
                # if step % params['eval_dev_batch'] == 0:
                #     K.set_learning_phase(0)
                #     dev_loss,dev_acc = do_eval(sess,cnn_rnn,dev_datas,batch_size,params['model_type'])
                #     K.set_learning_phase(1)
                #     timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
                #     str_loss = '{}: epoch:{},step:{},dev_loss:{},dev_acc:{}'.format(
                #         timestamp, epoch, step, dev_loss, dev_acc)
                #     print(str_loss)
                    # dev_writer.add_summary(
                    #     tf.Summary(value=[
                    #         tf.Summary.Value(tag="loss",simple_value=dev_loss),
                    #         tf.Summary.Value(tag="accuracy", simple_value=dev_acc),
                    #     ]),step)

                # 保存最好的四个模型
                # if dev_acc >= best_acc:
                #     best_acc = dev_acc
                #     saver.save(
                #         sess,
                #         model_path+'-%s'%best_acc,
                #         global_step=step,
                #         write_meta_graph=True
                #     )

if __name__ == '__main__':
    # load params
    params = yaml.load(open('utils/params_tf.yaml','r',encoding='utf-8'))

    train(params)
