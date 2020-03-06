import numpy as np
import tensorflow as tf
import math
import time
import datetime
import os
from tensorflow.python.framework import graph_util


def weight_variable(shape):
    tf.set_random_seed(1)
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(z):
    return tf.nn.max_pool(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def random_mini_batches(X, Y, mini_batch_size=16, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, int(num_complete_minibatches)):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def cnn_model(X_train, y_train, X_test, y_test,
              keep_prob, learning_rate=1e-4, l2_regularizer=5e-4,
              num_epochs=500, save_epoch=500,
              minibatch_size=16, model_path='./'
              ):
    X = tf.placeholder(tf.float32, [None, 64, 64, 3], name="input_x")
    y = tf.placeholder(tf.float32, [None, 11], name="input_y")
    kp = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")
    # lam = tf.placeholder(tf.float32, name="lamda")

    # conv1
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    z1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
    maxpool1 = max_pool_2x2(z1)  # max_pool1完后maxpool1维度为[?,32,32,32]

    # conv2
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    z2 = tf.nn.relu(conv2d(maxpool1, W_conv2) + b_conv2)
    maxpool2 = max_pool_2x2(z2)  # max_pool2,shape [?,16,16,64]

    # conv3  效果比较好的一次模型是没有这一层，只有两次卷积层，隐藏单元100，训练20次
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    z3 = tf.nn.relu(conv2d(maxpool2, W_conv3) + b_conv3)
    maxpool3 = max_pool_2x2(z3)  # max_pool3,shape [?,8,8,128]

    # full connection1
    W_fc1 = weight_variable([8 * 8 * 128, 100])
    b_fc1 = bias_variable([100])
    maxpool2_flat = tf.reshape(maxpool3, [-1, 8 * 8 * 128])
    z_fc1 = tf.nn.relu(tf.matmul(maxpool2_flat, W_fc1) + b_fc1)
    z_fc1_drop = tf.nn.dropout(z_fc1, keep_prob=kp)

    # softmax layer
    W_fc2 = weight_variable([100, 11])
    b_fc2 = bias_variable([11])
    z_fc2 = tf.add(tf.matmul(z_fc1_drop, W_fc2), b_fc2, name="outlayer")
    prob = tf.nn.softmax(z_fc2, name="probability")

    # cost function
    regularizer = tf.contrib.layers.l2_regularizer(l2_regularizer)
    regularization = regularizer(W_fc1) + regularizer(W_fc2)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z_fc2)) + regularization

    # 创建全局tensor
    global_steps = tf.train.create_global_step()

    # Learning rate
    learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_steps,
                                               decay_steps=200, decay_rate=0.1, staircase=False, name='learning_rate')

    # AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='AdamOptimizer')
    # Train option
    train_op = optimizer.minimize(loss=cost, global_step=global_steps)
    # train = tf.train.AdamOptimizer().minimize(cost, global_step=global_step)
    # output_type='int32', name="predict"
    pred = tf.argmax(prob, 1, output_type="int32", name="predict")  # 输出结点名称predict方便后面保存为pb文件
    correct_prediction = tf.equal(pred, tf.argmax(y, 1, output_type='int32'))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.set_random_seed(1)  # to keep consistent results

    seed = 0

    # 保存模型路径
    output_path = os.path.join(model_path, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print('Create the output dir in: ', str(output_path))

    ckpt_file = os.path.join(output_path, 'gusture')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, num_epochs + 1):
            seed = seed + 1
            epoch_cost = 0.
            num_minibatches = int(X_train.shape[0] / minibatch_size)
            minibatches = random_mini_batches(X_train, y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([train_op, cost],
                                             feed_dict={X: minibatch_X, y: minibatch_Y, kp: keep_prob})
                epoch_cost += minibatch_cost / num_minibatches
            if epoch % 10 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                print(str((time.strftime('%Y-%m-%d %H:%M:%S'))))
            if epoch % save_epoch == 0:
                # save model
                saver = tf.train.Saver({'W_conv1': W_conv1, 'b_conv1': b_conv1, 'W_conv2': W_conv2, 'b_conv2': b_conv2,
                                        'W_conv3': W_conv3, 'b_conv3': b_conv3,
                                        'W_fc1': W_fc1, 'b_fc1': b_fc1, 'W_fc2': W_fc2, 'b_fc2': b_fc2})
                saver.save(sess, ckpt_file, global_step=global_steps)
                print('Saving checkpoint-%d file' % epoch)
                print("Global_step_train: ", sess.run(global_steps))

        # 这个accuracy是前面的accuracy，tensor.eval()和Session.run区别很小
        train_acc = accuracy.eval(feed_dict={X: X_train[:1000], y: y_train[:1000], kp: keep_prob})
        # train_acc = accuracy.eval(feed_dict={X: X_train[:1000], y: y_train[:1000], kp: keep_prob, lam: lamda})
        print("train accuracy", train_acc)
        test_acc = accuracy.eval(feed_dict={X: X_test[:1000], y: y_test[:1000]})
        # test_acc = accuracy.eval(feed_dict={X: X_test[:1000], y: y_test[:1000], lam: lamda})
        print("test accuracy", test_acc)

        # 将训练好的模型保存为.pb文件，方便在Android studio中使用
        # output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def,
        #                                                              output_node_names=['predict'])
        # with tf.gfile.FastGFile('model/digital_gesture.pb',
        #                         mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
        #     f.write(output_graph_def.SerializeToString())
