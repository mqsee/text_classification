# -*- coding: utf-8 -*-
# @Time    : 2017/10/22 下午5:03
# @Author  : Qi MO
# @Contact  : putao0124@gmail.com
# @Site  : http://blog.csdn.net/coraline_m
# @File    : train_keras.py
# @Software: PyCharm

from utils.cnn_rnn_keras import CNNRNN
from keras.utils import plot_model
from utils.data_helper import load_data_cv
import yaml
import json

def train(params):
    """
    :param params:
    :return:
    """
    datas, voc_size = load_data_cv(
        file_path='../docs/raw/Train.csv',
        voc_dir='../docs/pro/',
        voc_mode=1 if params['model_type'] == 'char' else 2,  # 如果使用词模型 params.yaml文件中model_type改为word
        cv=10)
    params['vocab_size'] = voc_size
    if params['model_type'] == 'char':
        params['user_q_dim'] = len(datas[0].char_user_q)
    else:
        params['user_q_dim'] = len(datas[0].word_user_q)
    # indent=4 格式化输出，check params
    print(json.dumps(params, indent=4))

    model = CNNRNN(params).model

    dev_datas = list(filter(lambda data: data.cv == 1, datas))
    train_datas = list(filter(lambda data: data.cv != 1, datas))
    print('len train datas: {}'.format(len(train_datas)))
    print('len dev datas: {}'.format(len(dev_datas)))

    if params['model_type'] == 'char':  # 字模型
        train_X = [QM.char_user_q for QM in train_datas]
        dev_X = [QM.char_user_q for QM in dev_datas]
    else:  # 词模型
        train_X = [QM.word_user_q for QM in train_datas]
        dev_X = [QM.word_user_q for QM in dev_datas]
    train_Y = [QM.labels for QM in train_datas]
    dev_Y = [QM.labels for QM in dev_datas]

    hist = model.fit(
        x = train_X,
        y = train_Y,
        epochs=params['epoch'],
        batch_size=params['batch_size'],
        verbose=2,  # 2 for one log line per epoch
        validation_data=(dev_X, dev_Y))

    res = {}
    res['train_acc'] = hist.history['acc']
    res['val_acc'] = hist.history['val_acc']
    res['train_loss'] = hist.history['loss']
    res['val_loss'] = hist.history['val_loss']
    res['epoch'] = [epoch for epoch in range(1, params['epoch'] + 1)]

    model_name = '{}/{}_{}_{}_v2.h5'.format(params['model_dir'],params['filters'],params['pool_size_1'],params['pool_size_2'])
    train_path = '{}/{}_{}_{}_v2.json'.format(params['model_dir'],params['filters'],params['pool_size_1'],params['pool_size_2'])
    model_plot = '{}/{}_{}_{}_v2.png'.format(params['model_dir'],params['filters'],params['pool_size_1'],params['pool_size_2'])
    with open(train_path, 'w') as f:
        json.dump(res, f)

    plot_model(model, to_file=model_plot)
    model.save(model_name)

if __name__ == '__main__':
    # load params
    params = yaml.load(open('utils/params_keras.yaml', 'r', encoding='utf-8'))

    train(params)