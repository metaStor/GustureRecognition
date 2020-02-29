#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : img_to_h5.py
# @Author: ShenHao
# @Contact : 1427662743@qq.com
# @Date  : 20-3-1上午11:03
# @Desc  : 读取压缩后的图片, 存储为h5缓存文件


import os
from PIL import Image
import numpy as np
import h5py


# 图片转h5文件
def image_to_h5():
    dirs = os.listdir("resized_img")
    Y = []  # label
    X = []  # data
    print(len(dirs))
    for filename in dirs:
        label = int(filename.split('_')[0])
        Y.append(label)
        im = Image.open("resized_img/{}".format(filename)).convert('RGB')
        mat = np.asarray(im)  # image 转矩阵
        X.append(mat)

    if not os.path.exists('dataset'):
        os.mkdir('dataset')

    file = h5py.File("dataset/data.h5", "w")
    file.create_dataset('X', data=np.array(X))
    file.create_dataset('Y', data=np.array(Y))
    file.close()


# test
# data = h5py.File("dataset//data.h5","r")
# X_data = data['X']
# print(X_data.shape)
# Y_data = data['Y']
# print(Y_data[123])
# image = Image.fromarray(X_data[123]) #矩阵转图片并显示
# image.show()


if __name__ == "__main__":
    image_to_h5()
