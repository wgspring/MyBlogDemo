"""
预测酸奶的销量
标准值：y = x1 + x2
"""

import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
STEPS = 3000

# 数据集
X = np.random.rand(32, 2)
Y_ = [[(x1 + x2 + np.random.rand() / 10 - 0.05)] for (x1, x2) in X]

# 前向传播
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1]))
y = tf.matmul(x, w1)

# 后向传播
loss = tf.reduce_mean(tf.square(y - y_))  # 定义损失函数
train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 训练
with tf.Session() as sses:
    sses.run(tf.global_variables_initializer())
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sses.run(train, feed_dict={x: X[start:end], y_: Y_[start:end]})

        if i % 500 == 0:
            print("训练中 loss = ", sses.run(loss, feed_dict={x: X, y_: Y_}))

    # 测试成果
    print("w1 = ", sses.run(w1))
    result = sses.run(y, feed_dict={x: X})
    print(np.hstack((Y_, result)))
