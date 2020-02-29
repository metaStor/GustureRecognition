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


# 压缩图片,把图片压缩成64*64的
def resize_img():
    dirs = os.listdir("split_pic//6")
    for filename in dirs:
        im = tf.gfile.FastGFile("split_pic//6//{}".format(filename), 'rb').read()
        # print("正在处理第%d张照片"%counter)
        with tf.Session() as sess:
            img_data = tf.image.decode_jpeg(im)
            image_float = tf.image.convert_image_dtype(img_data, tf.float32)
            resized = tf.image.resize_images(image_float, [64, 64], method=3)
            resized_im = resized.eval()
            # new_mat = np.asarray(resized_im).reshape(1, 64, 64, 3)
            scipy.misc.imsave("resized_img6//{}".format(filename), resized_im)


if __name__ == "__main__":
    print("start.....: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
    resize_img()
    print("end....: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
