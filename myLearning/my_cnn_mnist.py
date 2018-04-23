#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:33:37 2018

@author: ming
"""

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data 
mnist = input_data.read_data_sets("/Users/ming/Documents/AI/Data/04-mnist/mnist", one_hot=True)



"""
为了创建这个模型，我们需要创建大量的权重和偏置项。
这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。
由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，
以避免神经元节点输出恒为0的问题（dead neurons）。为了不在建立模型的时候反复做初始化操作，
我们定义两个函数用于初始化。
truncated_normal：正态分布随机值。truncated：截断
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
#[batch, in_height, in_width, in_channels] 即[batch尺寸,横滑步长,竖滑步长,通道数]      SAME：自动补充
def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
   

#输入原始的数据占位
x = tf.placeholder("float", [None, 784])
#为了计算交叉熵，我们首先需要添加一个新的占位符用于：输入正确值
y_ = tf.placeholder("float", [None, 10])



#第一层卷积
"""
现在我们可以开始实现第一层了。它由一个卷积接一个max pooling完成。
卷积在每个5x5的patch中算出32个特征。卷积的权重张量形状是[5, 5, 1, 32]，
前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。 
而对于每一个输出通道都有一个对应的偏置量。
前两个参数是filter大小、第三个参数是原始图像通道数、第四个参数是‘得到的特征图’数量

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
"""
为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。
第三个参数：32，是第一层的输出大小
"""
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#前面的卷积已经将特征提取出来了，但是还没有利用上，下面就是使用
#全连接层
"""
经过两次pooling后，28*28 -> 14*14 -> 7*7 
图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
"""
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#Dropout:防止过拟合
"""
为了减少过拟合，我们在输出层之前加入dropout。
我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 
TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。
所以用dropout的时候可以不用考虑scale。
全连接时，屏蔽一些，不用全部参与，keep_prob参与比例，1代表全部参与。使得一些神经元不参与训练。
"""
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#输出层(第二个全连接层):添加softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)





"""
--------------------------------------------------------------------------
--------------------->训练和评估模型<----------------------
--------------------------------------------------------------------------


为了进行训练和评估，我们使用与之前简单的单层SoftMax神经网络模型几乎相同的一套代码，
只是我们会用更加复杂的ADAM优化器来ne做梯度最速下降，在feed_dict中加入额外的参数keep_prob来控制dropout比例。
然后每100次迭代输出一次日志。
唐老师demo:
cross_entropy = -tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y))
"""

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


#在运行计算之前，我们需要添加一个操作来初始化我们创建的变量。
init = tf.global_variables_initializer()
#现在我们可以在一个Session里面启动我们的模型，并且初始化变量
sess = tf.Session()
sess.run(init)



for i in range(200):
    batch = mnist.train.next_batch(50)
    
    if i%100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})    

print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
