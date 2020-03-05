#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py.py
# @Author: ShenHao
# @Contact : 1427662743@qq.com 
# @Date  : 20-3-1上午10:26
# @Desc  : 加载数据和网络, 训练模型


from net.cnn import *
from load_data import *


if __name__ == "__main__":
    print("载入数据集: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
    X_train, X_test, y_train, y_test = load_dataset()
    print("开始训练: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
    cnn_model(X_train, y_train, X_test, y_test,
              keep_prob=0.5, learning_rate=1e-4, l2_regularizer=5e-4,
              num_epochs=2000, save_epoch=5000, minibatch_size=16,
              model_path='/media/meta/Work/Study_and_Work/Graduation/GustureRecognition/dataSet/models')
    print("训练结束: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
