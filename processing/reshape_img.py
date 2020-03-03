#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : reshape_img.py
# @Author: ShenHao
# @Contact : 1427662743@qq.com 
# @Date  : 20-2-29下午11:03
# @Desc  : 压缩图片,把图片压缩成64*64的

import os
import tensorflow as tf
import scipy.misc
import time


# 分批处理 防止由于内存不够而宕机
src = r'/media/meta/Free time/gene_pic/10/10_3'
out = r'/media/meta/Work/Study_and_Work/Graduation/GustureRecognition/dataSet/Images/generate_pic'


# 压缩图片,把图片压缩成64*64的
def resize_img():
    dirs = os.listdir(src)
    counter = 0
    with tf.Session() as sess:
        for filename in dirs:
            im = tf.gfile.FastGFile(os.path.join(src, filename), 'rb').read()
            print("%d\t正在处理图片: %s" % (counter, filename))
            img_data = tf.image.decode_jpeg(im)
            image_float = tf.image.convert_image_dtype(img_data, tf.float32)
            resized = tf.image.resize_images(image_float, [64, 64], method=3)
            resized_im = resized.eval()
            # new_mat = np.asarray(resized_im).reshape(1, 64, 64, 3)
            scipy.misc.imsave(os.path.join(out, filename), resized_im)
            counter += 1


if __name__ == "__main__":
    print("Start at " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
    resize_img()
    print("End at " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
