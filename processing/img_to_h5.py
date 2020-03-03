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


img_dir = r'../dataSet/Images/generate_pic'
out_h5 = r'../dataSet/cache'


# 图片转h5文件
def image_to_h5():
    dirs = os.listdir(img_dir)
    Y = []  # label
    X = []  # data
    print(len(dirs))
    for filename in dirs:
        label = int(filename.split('_')[0])
        Y.append(label)
        im = Image.open(os.path.join(img_dir, filename)).convert('RGB')
        mat = np.asarray(im)  # image 转矩阵
        X.append(mat)

    file = h5py.File(os.path.join(out_h5, "data.h5"), "w")
    file.create_dataset('X', data=np.array(X))
    file.create_dataset('Y', data=np.array(Y))
    file.close()


if __name__ == "__main__":
    image_to_h5()
