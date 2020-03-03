#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_augmentation2.py
# @Author: ShenHao
# @Contact : 1427662743@qq.com
# @Date  : 20-2-27上午12:11
# @Desc  : 生成数据, 每一张图片经过旋转，平移，拉伸 等操作每张图片生成若干张


from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

dirs = r'/media/meta/Work/Study_and_Work/Graduation/GustureRecognition/dataSet/Images/picture/9'
output = r'/media/meta/Free time/gene_pic'
# output = r'/media/meta/Work/Study_and_Work/Graduation/GustureRecognition/dataSet/Images/generater_pic'
Gene = 10

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

print('Augmenting class: %s' % dirs)
for filename in sorted(os.listdir(dirs)):
    file = os.path.join(dirs, filename)
    img = load_img(file)
    x = img_to_array(img)
    # print(x.shape)
    x = x.reshape((1,) + x.shape)  # datagen.flow要求rank为4
    # print(x.shape)
    datagen.fit(x)
    prefix = filename.split('.')[0]
    print('\tGenerate %s' % prefix, end='\t')
    counter = 0
    for batch in datagen.flow(x, batch_size=2, save_to_dir=output, save_prefix=prefix, save_format='jpg'):
        counter += 1
        if counter > Gene:
            break  # 达到一定次数, 退出循环
    os.remove(file)  # remove source image
    print('Finished, Removing it ...')

print('Done!')
