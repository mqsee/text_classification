# -*- coding: utf-8 -*-
# @Time    : 2017/10/20 下午12:28
# @Author  : Qi MO
# @Contact  : putao0124@gmail.com
# @Site  : http://blog.csdn.net/coraline_m
# @File    : pre_process.py
# @Software: PyCharm

import json
import pandas as pd

file_dir = '../../docs/raw/'

# 得到同义词词典
change_f = open('{}同义词库1.utf8'.format(file_dir),encoding='utf-8')  # 返回一个文件对象
change_dict = {}
while 1:
    line = change_f.readline()  # 调用文件的 readline()方法
    line = line.strip('\n')
    if not line:
        break
    wordlist = line.split(',')
    for word in wordlist:
        change_dict[word] = wordlist[0]

# 得到停用词
stop_f = open('{}停用词_简版.utf8'.format(file_dir),encoding='utf-8')
stop_dict = {}
while 1:
    line = stop_f.readline()  # 调用文件的 readline()方法
    line = line.strip('\n')
    if not line:
        break
    stop_dict[line] = 1

change_stop_dict = {'change_dict':change_dict,'stop_dict':stop_dict}
change_stop_dict_path = '../../docs/pro/change_stop.json'
with open(change_stop_dict_path, 'w') as f:
    json.dump(change_stop_dict, f)

# 得到标准问题ID对应标签dict
std_label_dict = {}
df = pd.read_csv('{}Std_Q.csv'.format(file_dir))
print(df.head())
std_label_dict_path = '../../docs/pro/std_label.json'
for std_id,labels in zip(df['std_id'],df['labels']):
    std_label_dict[str(std_id)] = int(labels)
with open(std_label_dict_path, 'w') as f:
    json.dump(std_label_dict, f)
