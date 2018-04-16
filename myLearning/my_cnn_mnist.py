#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:33:37 2018

@author: ming
"""

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data 
mnist = input_data.read_data_sets("/Users/ming/Documents/AI/Data/04-mnist/mnist", one_hot=True)

x = tf.placeholder("float", [None, 784])
#W = tf.Variable(tf.zeros([784,10]))
#b = tf.Variable(tf.zeros([10]))


"""
为了创建这个模型，我们需要创建大量的权重和偏置项。
这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。
由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，
以避免神经元节点输出恒为0的问题（dead neurons）。为了不在建立模型的时候反复做初始化操作，
我们定义两个函数用于初始化。
"""

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

"""
TensorFlow在卷积和池化上有很强的灵活性。
我们怎么处理边界？步长应该设多大？在这个实例里，我们会一直使用vanilla版本。
我们的卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小。
我们的池化用简单传统的2x2大小的模板做max pooling。为了代码更简洁，我们把这部分抽象成一个函数。
"""

#卷积和池化
#[batchSize,横滑步长,竖滑步长,channelSize]      SAME：自动补充
def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
   
    
#第一层卷积
"""
现在我们可以开始实现第一层了。它由一个卷积接一个max pooling完成。
卷积在每个5x5的patch中算出32个特征。卷积的权重张量形状是[5, 5, 1, 32]，
前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。 
而对于每一个输出通道都有一个对应的偏置量。
"""

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

"""
为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，
最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
第一维-1代表取合适值，即实际的图片数量：x = tf.placeholder("float", [None, 784])
"""
x_image = tf.reshape(x, [-1, 28, 28, 1])

#我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#第二层卷积
"为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。"
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#经过两次pooling后，28*28 -> 14*14 -> 7*7
