# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import cv2
def imReadHW(impath):
    img = cv2.imread(impath)  # 406*1000*3
    print("=========某一行=========")
    print(img[0, :10, :])
    print(np.shape(img[0, :10, :]))
    hArr = img[0, :, :]  # 对RGB图像按行读取,有h行,每一行一个通道
    print("=========某一列=========")
    print(img[:10, 0, :])
    print(np.shape(img[:10, 0, :]))

def network_structure_CNN(value, cnn_in_batch,imgChannel,h,w):
    ## imput:图像矩阵、图像张数、图像通道数
    ## 生成卷积的输入: 输入batch量（多少张图片）、图高、图宽、图像通道数
    CNNtemp1 = tf.reshape(value, [cnn_in_batch, h, w, imgChannel])
    CNNtemp1 = tf.image.convert_image_dtype(CNNtemp1, tf.float32)
    ## 设置卷卷积核 核高、核宽、图像通道数、卷积核个数
    conv2d_filter_0 = tf.Variable(tf.random.normal([5, 3, imgChannel, 1]), dtype=tf.float32, name='conv2d_filter')
    ## 配置卷积 输入、卷积核、步长、卷积方式
    CNNtemp2_0 = tf.nn.conv2d(CNNtemp1, filter=conv2d_filter_0, strides=[1, 1, 1, 1], padding='VALID')  # SAME
    CNNtemp3 = tf.reshape(CNNtemp2_0, [cnn_in_batch, (h-5+1), (w-3+1)])  ## 元素个数 = 单个卷积结果*卷积核个数
    return CNNtemp3#tf.nn.relu6(CNNtemp3)

def imgOne(impath):
    img = cv2.imread(impath)  # 406*1000*3
    # 对RGB图像按行读取,有h行,每一行一个通道
    imgh = []
    for i in range(406):
        imgh.append(network_structure_CNN(img[i, :, :], 1, 1, 1000, 3))
    # 对RGB图像按列读取,有w列,每一列一个通道
    imgw = []
    for i in range(1000):
        imgw.append(network_structure_CNN(img[:, i, :], 1, 1, 406, 3))
    return imgh,imgw


def testCNN(impath):
    img = cv2.imread(impath)  # 406*1000*3
    print("===============", np.shape(img), "===============")
    imgh, imgw = imgOne(impath)
    imghw = network_structure_CNN(imgh, 1, 1, 406, 996)
    imgwh = network_structure_CNN(imgw, 1, 1, 1000, 402)

    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)

        resulth = sess.run(imgh)
        resultw = sess.run(imgw)
        cv2.imwrite("imgh.png", np.reshape(resulth, (406, 996, 1))*255)
        cv2.imwrite("imgw.png", np.reshape(resultw, (1000, 402, 1))*255)

        resulthw = sess.run(imghw)
        resultwh = sess.run(imgwh)
        cv2.imwrite("imghw.png", np.reshape(resulthw, (402, 994, 1)) * 255)
        cv2.imwrite("imgwh.png", np.reshape(resultwh, (996, 400, 1)) * 255)
        # print("=======imghw========", np.shape(resulthw), "===============")
        # print("=======imgwh========", np.shape(resultwh), "===============")
        # cv2.imshow("按行读取卷积，列会减少", np.reshape(resulthw, (402, 994, 1)))
        # cv2.waitKey(10000)
        # cv2.imshow("按列读取卷积，行会减少", np.reshape(resultwh, (996, 400, 1)))
        # cv2.waitKey(300000)
    print("=========================OVER=========================")



if __name__=="__main__":
    testCNN("cat.jpg")
    pass