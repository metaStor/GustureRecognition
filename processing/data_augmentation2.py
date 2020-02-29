#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_augmentation2.py
# @Author: ShenHao
# @Contact : 1427662743@qq.com
# @Date  : 20-2-27上午12:11
# @Desc  : 生成数据, 每一张图片经过旋转，平移，拉伸 等操作每张图片生成若干张


from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

Gene = 100

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


dirs = os.listdir("picture")
print(len(dirs))

for filename in dirs:
    img = load_img("picture/{}".format(filename))
    x = img_to_array(img)
    # print(x.shape)
    x = x.reshape((1,) + x.shape)  # datagen.flow要求rank为4
    # print(x.shape)
    datagen.fit(x)
    prefix = filename.split('.')[0]
    print(prefix)
    counter = 0
    for batch in datagen.flow(x, batch_size=4, save_to_dir='generater_pic', save_prefix=prefix, save_format='jpg'):
        counter += 1
        if counter > Gene:
            break  # 达到一定次数, 退出循环
