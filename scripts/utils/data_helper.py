# -*- coding: utf-8 -*-
# @Time    : 2017/10/19 下午7:21
# @Author  : Qi MO
# @Contact  : putao0124@gmail.com
# @Site  : http://blog.csdn.net/coraline_m
# @File    : data_helper.py
# @Software: PyCharm

import jieba
import json
import os
import pandas as pd
import numpy as np
from hanziconv import HanziConv

class QuestionMatch(object):
    def __init__(self,user_q,std_id,cv):
        """
        问题匹配对
        :param user_q:  用户问题
        :param std_id:  标准问题ID
        param std_id:  交叉验证标记
        """
        self.user_q = user_q # 用户问题
        self.std_id = std_id  # 标准问题ID
        self.cv = cv
        self.labels = [0]*3055  # 类别标签,one-hot表示，3055个类别
        self.deal_char_user_q = [] # 字符级的处理的用户问题,包括全角转半角、繁体转简体、手工去掉一些特殊符号
        self.deal_word_user_q = []  # 词级别的处理用户问题，包括去停用词、同义词替换等
        self.std_q = '' # 标准问题
        self.char_user_q = [] # 字符级用户问题, 用字符级的ID代替
        self.word_user_q = [] # 词级别用户问题, 用词级别的ID代替
        self.pre_std_id_list = [] # 模型预测的最匹配的标准问题ID
        self.pre_std_q_list = []  # 模型预测的最匹配标准问题
        self.pre_std_score_list = []  # 模型预测的最匹配标准问题的得分
        self.get_deal_char_user_q()

    def get_deal_char_user_q(self):
        """
        得到字符级的处理的用户问题，原始用户问题转换成list
        :return:
        """
        sentence = (str(self.user_q)).lower().strip()
        # 去掉标点符号，不包括+号
        punc_code = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 58, 59, 60, 61, 62, 63, 64, 91, 92, 93, 94,
                     95, 96, 123, 124, 125, 126]
        # 去掉特殊字符，包含日文，特殊符号等
        spe_code = [32, 12290, 12289, 8220, 8221, 183, 8230, 9, 8216, 12305, 8211, 12299, 8217, 32415, 215, 12298, 8561,
                    20008, 8857, 19973, 57430, 8593, 20101, 20022, 58386, 8560, 176, 8592, 58385, 20031, 8801, 8594,
                    57647, 58381, 9583, 9584, 8734, 969, 12297, 8470, 30103, 57607, 8743, 12296, 247, 9675, 711, 960,
                    9581, 9582, 9318, 58371, 8451, 9734, 57354, 12302, 12303, 713, 8600, 58382, 9317, 9312, 9733, 1072,
                    58163, 57352, 8730, 953, 23379, 57356, 12300, 12301, 9315, 9472, 8746, 57431, 8805, 8745, 12398,
                    1079, 8736, 8251, 21661, 12469, 12512, 12473, 12531, 12364, 12434, 12377, 35388, 20059, 65507, 9615,
                    20058, 8741, 12391, 945, 955, 65097, 20155, 9602, 12306, 13212, 20981, 58390, 12308, 12309, 952,
                    21241, 9650, 65123, 963, 9651, 65076, 950, 9587, 59416, 252, 8595, 35744, 59047, 58397, 237, 12540,
                    12291, 57606, 12583, 168, 12553, 22200, 58392, 9354, 12310, 12311, 167, 58400, 58372, 242, 58160,
                    65103, 9661, 8453, 57347, 9520, 57642, 57668, 58377, 9473, 8214, 956, 12525, 9670, 20886, 1094,
                    1074, 1087, 8208, 8242, 8599, 58389, 8757, 58388, 257]
        for x in sentence:
            x = HanziConv.toSimplified(x) # 繁简体转换
            inside_code = ord(x)
            # 全角转半角
            if inside_code == 12288:
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):
                inside_code -= 65248
            if inside_code not in punc_code and inside_code not in spe_code:
                self.deal_char_user_q.append(chr(inside_code))
        del sentence,punc_code,spe_code

    def get_char_user(self,char_voc,max_length):
        """
        得到字符级用户问题的表示，用ID代替，包括paddding
        :param char_voc:   字符级的词典
        :param max_length: padding长度
        :return:
        """
        deal_char_user_q = self.deal_char_user_q[:max_length]
        self.char_user_q = [char_voc[c] if c in char_voc else 0for c in deal_char_user_q]
        self.char_user_q.extend([0]*(max_length-len(deal_char_user_q)))
        del deal_char_user_q

    def get_deal_word_user_q(self,change_dict,stop_dict):
        """
        得到词级别的处理的用户问题，原始用户问题转换成list
        :return:
        """
        seg_words = [w for w in jieba.cut(self.user_q)]
        self.deal_word_user_q = [change_dict[w] if w in change_dict else w for w in seg_words if w not in stop_dict and w!= ' ']
        del seg_words

    def get_word_user(self,word_voc,max_length):
        """
        得到词级别级用户问题的表示，用ID代替，包括paddding
        :param char_voc:   词级别的词典
        :param max_length: padding长度
        :return:
        """
        deal_word_user_q = self.deal_word_user_q[:max_length]
        self.word_user_q = [word_voc[c] if c in word_voc else 0for c in deal_word_user_q]
        self.word_user_q.extend([0]*(max_length-len(deal_word_user_q)))
        del deal_word_user_q

    def get_labels(self,std_label_dict):
        """
        得到标签，one-hot向量表示
        :param std_label_dict: 标准问题ID对应的label
        :return:
        """
        self.labels[std_label_dict[str(self.std_id)]] = 1

def build_vocab(q_list,voc_path):
    """
    建立词典
    :param q_list:  二维数组，分好字或者分好词的数据
    :param voc_path:  词典地址
    :return:
    """
    print('build vocab...{}'.format(voc_path))
    voc = {'<s>':0}
    voc_index = 0
    max_len = 0
    for q in q_list:
        max_len = max(max_len,len(q))
        for x in q:
            if x not in voc:
                voc_index+=1
                voc[x] = voc_index
    print('the size of {} is {}'.format(voc_path,len(voc)))
    print('max_len is {}'.format(max_len))
    voc_dict = {'voc':voc,'max_len':max_len}
    with open(voc_path,'w') as f:
        json.dump(voc_dict,f)
    print('build vocab...{} done'.format(voc_path))
    return voc,max_len

def load_one(voc_dir,user_q):
    """
    构建一个QM对象，用来测试
    :param user_q:
    :return:
    """
    QM = QuestionMatch(user_q=user_q,std_id=-1,cv=-1)
    char_voc_path = '{}char_voc.json'.format(voc_dir)
    if os.path.isfile(char_voc_path):
        with open(char_voc_path, 'r') as f:
            char_voc_dir = json.load(f)
            char_voc = char_voc_dir['voc']
            char_max_len = char_voc_dir['max_len']
        QM.get_char_user(char_voc, char_max_len)
    return QM

def load_data_cv(file_path,voc_dir,voc_mode,cv=5):
    """
    :param file_path:  文件地址 train or test
    :param voc_dir:    字典的目录地址
    :param voc_mode:   字典模式，只有3种。1表示字模型，2表示词模型，3表示字和词都用
    :param cv:    几折交叉验证
    :return:
    """
    df = pd.read_csv(file_path)
    print('load data...{}'.format(file_path))
    rev = [] # 返回QuestionMath对象的list
    for user_q,std_id in zip(df['user_q'],df['std_id']):
        rev.append(QuestionMatch(user_q,std_id,np.random.randint(0, cv)))

    std_label_path = '{}std_label.json'.format(voc_dir)
    with open(std_label_path,'r') as f:
        std_label_dict = json.load(f)
    for QM in rev:
        QM.get_labels(std_label_dict)

    if voc_mode&1 : # 表示用字模型
        print('load data char user_q...')
        char_voc_path = '{}char_voc.json'.format(voc_dir)
        if os.path.isfile(char_voc_path):
            with open(char_voc_path,'r') as f:
                char_voc_dir = json.load(f)
                char_voc = char_voc_dir['voc']
                char_max_len = char_voc_dir['max_len']
        else:
            char_voc,char_max_len = build_vocab([x.deal_char_user_q for x in rev],char_voc_path)
        for QM in rev:
            QM.get_char_user(char_voc, char_max_len)

    if voc_mode&2: # 表示用词模型
        print('load data word user_q...')
        word_voc_path = '{}word_voc.json'.format(voc_dir)
        # 同义词和停用词
        change_stop_dict_path = '{}change_stop.json'.format(voc_dir)
        with open(change_stop_dict_path,'r') as f:
            change_stop_dict = json.load(f)
        change_dict, stop_dict  = change_stop_dict['change_dict'],change_stop_dict['stop_dict']
        # 加载用户词典
        user_dict_path = '{}用户词典2.utf8'.format(voc_dir)
        jieba.load_userdict(user_dict_path)
        for QM in rev:
            QM.get_deal_word_user_q(change_dict,stop_dict)
        if os.path.isfile(word_voc_path):
            with open(word_voc_path, 'r') as f:
                word_voc_dir = json.load(f)
                word_voc = word_voc_dir['voc']
                word_max_len = word_voc_dir['max_len']
        else:
            word_voc, word_max_len = build_vocab([x.deal_word_user_q for x in rev],word_voc_path)
        for QM in rev:
            QM.get_word_user(word_voc, word_max_len)

    # 如果内存不够用的话，可以把deal_char_user_q和deal_word_user_q置空
    # for QM in rev:
    #     print(QM.user_q, QM.std_id, QM.deal_char_user_q, QM.char_user_q)
    #     print(QM.user_q, QM.std_id, QM.deal_word_user_q, QM.word_user_q)
    # print(QM.labels)
    return rev, len(char_voc) if voc_mode == 1 else len(word_voc)

if __name__ == '__main__':
    train_path = '../../docs/raw/Train_1000.csv'
    test_path = '../../docs/raw/Test_1000.csv'
    voc_dir = '../../docs/pro/'
    voc_mode = 1
    rev,voc_size = load_data_cv(train_path,voc_dir,voc_mode)
    rev,voc_size = load_data_cv(test_path, voc_dir, voc_mode)