#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : load_data.py
# @Author: ShenHao
# @Contact : 1427662743@qq.com 
# @Date  : 20-3-1上午10:26
# @Desc  : 加载图片数据


import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils


# load dataset
def load_dataset(cache):
    print('Loading dataset in %s ...' % cache)
    # 划分训练集、测试集
    data = h5py.File(cache, "r")
    X_data = np.array(data['X'])  # data['X']是h5py._hl.dataset.Dataset类型，转化为array
    Y_data = np.array(data['Y'])
    # print(type(X_data))
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, train_size=0.9, test_size=0.1, random_state=22)
    # print(X_train.shape)
    # print(y_train[456])
    # image = Image.fromarray(X_train[456])
    # image.show()
    # y_train = y_train.reshape(1,y_train.shape[0])
    # y_test = y_test.reshape(1,y_test.shape[0])
    print(X_train.shape)
    # print(X_train[0])
    X_train = X_train / 255.  # 归一化
    X_test = X_test / 255.
    # print(X_train[0])
    # one-hot
    y_train = np_utils.to_categorical(y_train, num_classes=11)
    print(y_train.shape)
    y_test = np_utils.to_categorical(y_test, num_classes=11)
    print(y_test.shape)

    return X_train, X_test, y_train, y_test
