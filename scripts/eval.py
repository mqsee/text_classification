# -*- coding: utf-8 -*-
# @Time    : 2017/10/21 下午12:55
# @Author  : Qi MO
# @Contact  : putao0124@gmail.com
# @Site  : http://blog.csdn.net/coraline_m
# @File    : eval.py
# @Software: PyCharm

import numpy as np

def do_eval(sess,model,eval_data,batch_size,model_type):
    """
    eval dev/test data for model
    :param sess:
    :param model:
    :param eval_data:
    :param batch_size:
    :return:
    """
    number_of_data = len(eval_data)
    labels,probs = [],[]
    eval_loss,eval_cnt = 0.,0

    for start in range(number_of_data):
        end = start+batch_size
        if model_type == 'char':  # 字模型
            user_q = [QM.char_user_q for QM in eval_data[start:end]]
        else:  # 词模型
            user_q = [QM.word_user_q for QM in eval_data[start:end]]
        cur_labels = [QM.labels for QM in eval_data[start:end]]

        cur_loss,cur_probs = sess.run(
            [model.loss,model.probs],
            feed_dict = {
                model.user_q: user_q,
                model.labels: cur_labels
            })
        eval_loss += cur_loss
        eval_cnt += 1
        labels.append(cur_labels)
        probs.append(cur_probs)

    labels = np.array(labels)
    probs = np.array(probs)
    acc = np.mean(np.argmax(labels,axis=1)==np.argmax(probs,axis=1))
    return eval_loss/eval_cnt,acc