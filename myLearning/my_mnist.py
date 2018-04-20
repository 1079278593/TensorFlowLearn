#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:29:00 2018

@author: ming
"""

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data 
mnist = input_data.read_data_sets("/Users/ming/Documents/AI/Data/04-mnist/mnist", one_hot=True)

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#定义模型
y = tf.nn.softmax(tf.matmul(x,W) + b)

"""接下来要评估模型：我们通常定义指标来表示一个模型是坏的，这个指标称为成本（cost）或损失（loss），
然后尽量最小化这个指标。
一个非常常见的，非常漂亮的成本函数是“交叉熵”（cross-entropy）。
交叉熵产生于信息论里面的信息压缩编码技术，但是它后来演变成为从博弈论到机器学习等其他领域里的重要技术手段。

加入一个额外的偏置量（bias），因为输入往往会带有一些无关的干扰量

------------------------------------------------------------------------------------
------->y 是我们预测的概率分布, y' 是实际的分布（我们输入的one-hot vector)。<--------
------------------------------------------------------------------------------------



比较粗糙的理解是，交叉熵是用来衡量我们的预测用于描述真相的低效性
"""

#为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值
y_ = tf.placeholder("float", [None, 10])


"""
首先，用 tf.log 计算 y 的每个元素的对数。接下来，我们把 y_ 的每一个元素和 tf.log(y_) 的对应元素相乘。
最后，用 tf.reduce_sum 计算张量的所有元素的总和。
（注意，这里的交叉熵不仅仅用来衡量单一的一对预测和真实值，而是所有100幅图片的交叉熵的总和。
对于100个数据点的预测表现比单一数据点的表现能更好地描述我们的模型的性能。
"""
#计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

"""
现在我们知道我们需要我们的模型做什么啦，用TensorFlow来训练它是非常容易的。
因为TensorFlow拥有一张描述你各个计算单元的图，
它可以自动地使用反向传播算法(backpropagation algorithm)来有效地确定你的
变量是如何影响你想要最小化的那个成本值的。然后，TensorFlow会用你选择的优化算法
来不断地修改变量以降低成本。
"""

#在这里，我们要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#在运行计算之前，我们需要添加一个操作来初始化我们创建的变量。
#init = tf.initialize_all_variables()早期官方教程代码，需要改成下面的
init = tf.global_variables_initializer()

#现在我们可以在一个Session里面启动我们的模型，并且初始化变量
#变量需要通过seesion初始化后，才能在session中使用
#在初次调用时，init操作只包含了变量初始化程序tf.group。图表的其他部分不会在这里，而是在下面的训练循环运行。
sess = tf.Session()
sess.run(init)

#开始训练模型，模型循环训练1000次！
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs,y_:batch_ys})
    
#评估我们的模型
"""
tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，比如tf.argmax(y,1)返回的
是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签，我们可以
用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。

这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。
例如，[True, False, True, True] 经过cast的转换会变成 [1,0,1,1] ，取平均值后得到 0.75

"""
correct_predicton = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_predicton, "float"))

#此时run的数据是test
print(sess.run(accuracy, feed_dict={x:mnist.test.images,y_:mnist.test.labels}))