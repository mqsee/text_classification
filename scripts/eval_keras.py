# -*- coding: utf-8 -*-
# @Time    : 2017/10/22 下午11:54
# @Author  : Qi MO
# @Contact  : putao0124@gmail.com
# @Site  : http://blog.csdn.net/coraline_m
# @File    : eval_keras.py
# @Software: PyCharm

from keras.models import load_model
from utils.data_helper import load_data_cv
from utils.data_helper import load_one
import yaml
import json
import numpy as np
import pandas as pd
params = yaml.load(open('utils/params_keras.yaml', 'r', encoding='utf-8'))

def test_file(test_path, topk=5):
    """
    :param file_name:
    :return:
    """
    # load_test_data
    voc_dir = '../docs/pro/'
    voc_mode = 1 if params['model_type'] == 'char' else 2
    datas, voc_size = load_data_cv(test_path, voc_dir, voc_mode)
    if params['model_type'] == 'char':  # 字模型
        test_X = [QM.char_user_q for QM in datas]
    else:  # 词模型
        test_X = [QM.word_user_q for QM in datas]
    test_Y = [QM.labels for QM in datas]

    # load model
    # model_path = '../docs/keras/100_2_3.h5'
    model_path = '../docs/keras/100_2_3.h5'
    model = load_model(model_path)

    # predict
    predict_Y = model.predict(test_X, batch_size=1024)
    accuracy = np.mean(np.argmax(test_Y,axis=1)==np.argmax(predict_Y,axis=1))
    print('the test size of {} is {}'.format(test_path, len(datas)))
    print('the accuracy of {} is {}'.format(test_path,accuracy))

    # 标签到标准问题ID的dict, 标准问题ID到标准问题的dict
    label_id_dict, id_std_dict = {},{}
    std_q_path = '../docs/raw/Std_Q.csv'
    df_std = pd.read_csv(std_q_path)
    for std_id,std_q,labels in zip(df_std['std_id'],df_std['std_q'],df_std['labels']):
        label_id_dict[labels] = std_id
        id_std_dict[std_id] = std_q

    # 得到topk的预测问题ID, 预测问题, 预测问题得分
    for QM,predict_y in zip(datas,predict_Y):
        predict = predict_y.tolist()
        QM.std_q = id_std_dict[QM.std_id]
        for i in range(topk):
            max_value = max(predict)
            max_index = predict.index(max_value)
            predict[max_index] = -1.
            QM.pre_std_score_list.append(max_value)
            QM.pre_std_id_list.append(label_id_dict[max_index])
            QM.pre_std_q_list.append(id_std_dict[label_id_dict[max_index]])

    # 得到topk的acc, 不考虑top阈值
    topn_acc = [0]*topk
    for QM in datas:
        for k in range(topk):
            if QM.std_id in QM.pre_std_id_list[:k+1]:
                topn_acc[k]+=1.
    topn_acc = [count/len(datas) for count in topn_acc]
    print('the acc of top {} is {}'.format(topk,topn_acc))

    # 得到top3的acc, 考虑top阈值
    top_threshold = 0.46
    top1_ok,top1,top3_ok,top3 = 0,0,0,0
    for QM in datas:
        if QM.pre_std_score_list[0] - QM.pre_std_score_list[1] > top_threshold:
            top1+=1.
            top1_ok+=1. if QM.std_id == QM.pre_std_id_list[0] else 0
        else:
            top3+=1.
            top3_ok+=1. if QM.std_id in QM.pre_std_id_list[:3] else 0
    print('the threshold is {}'.format(top_threshold))
    print('the amount_percentage and acc of top1 is {} and {}'.format(top1/len(datas),top1_ok/top1))
    print('the amount_percentage and acc of top3 is {} and {}'.format(top3 / len(datas), top3_ok / top3))

def test_one(model,user_q):
    """
    返回topk接口
    :param model:  模型
    :param user_q: 用户问题
    :return:
    """

    voc_dir = '../docs/pro/'
    # 构建QuestionMatch类，并开始预测
    QM = load_one(voc_dir,user_q)
    test_X = np.array([QM.char_user_q])
    predict_Y = model.predict(test_X)

    # 标签到标准问题ID的dict, 标准问题ID到标准问题的dict
    label_id_dict, id_std_dict = {}, {}
    std_q_path = '../docs/raw/Std_Q.csv'
    df_std = pd.read_csv(std_q_path)
    for std_id, std_q, labels in zip(df_std['std_id'], df_std['std_q'], df_std['labels']):
        label_id_dict[labels] = std_id
        id_std_dict[std_id] = std_q

    # 得到top3的标准问题ID, 标准问题, 标准问题得分
    topk = 3
    predict = predict_Y[0].tolist()
    for i in range(topk):
        max_value = max(predict)
        max_index = predict.index(max_value)
        predict[max_index] = -1.
        QM.pre_std_score_list.append(max_value)
        QM.pre_std_id_list.append(label_id_dict[max_index])
        QM.pre_std_q_list.append(id_std_dict[label_id_dict[max_index]])
    # print('user_q: {}'.format(QM.user_q))
    # print('std_id: {}'.format(QM.pre_std_id_list))
    # print('std_q: {}'.format(QM.pre_std_q_list))
    # print('std_score: {}'.format(QM.pre_std_score_list))

    # 返回的dict
    rev = {}
    rev['user_q'] = user_q
    rev['top'] = 1
    rev['result'] = []
    top_threshold = 0.46
    if QM.pre_std_score_list[0] - QM.pre_std_score_list[1] < top_threshold:
        rev['top'] = topk

    for i in range(rev['top']):
        q_dict = {}
        q_dict['q_id'] = int(QM.pre_std_id_list[i])
        q_dict['score'] = QM.pre_std_score_list[i]
        q_dict['standard_q'] = QM.pre_std_q_list[i]
        rev['result'].append(q_dict)
    rev['seg'] = QM.deal_char_user_q

    return rev

if __name__ == '__main__':
    model_path = '../docs/keras/100_2_3.h5'
    model = load_model(model_path)
    user_q = 's7edge什么时候可以更新7.0'
    print(test_one(model,user_q))
    # test_path = '../docs/raw/Test_A.csv'
    # test_file(test_path,5)
    # test_path = '../docs/raw/Test_B.csv'
    # test_file(test_path,5)
