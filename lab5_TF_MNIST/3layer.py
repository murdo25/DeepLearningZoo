import matplotlib.pyplot as blt
import tensorflow as tf
import time
start_time = time.time()

# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist= input_data.read_data_sets("MNIST_data/", one_hot=True) 

#Define Graph
x1 = tf.placeholder(tf.float32, [None,784])
W1 = tf.Variable(tf.random_normal([784,784],stddev=0.001),name='W1')
b1 = tf.Variable(tf.random_normal([784]))
x2 = tf.placeholder(tf.float32, [None,784])
W2 = tf.Variable(tf.random_normal([784,784],stddev=0.001),name='W1')
b2 = tf.Variable(tf.random_normal([784]))
x3 = tf.placeholder(tf.float32, [None,784])
W3 = tf.Variable(tf.random_normal([784,10],stddev=0.001),name='W2')
b3 = tf.Variable(tf.random_normal([10]))

#hidden layer 1
x2 = tf.add(tf.matmul(x1,W1), b1)


#fully connected hidden layer 2
x2 = tf.nn.relu(x2)
x3 = tf.add(tf.matmul(x2,W2),b2)

#feed forward hidden layer 3
out = tf.add(tf.matmul(x3,W3),b3)
y = tf.nn.softmax(out)

#Cross Entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#Train!
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)


#init?
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
acc = []

for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x1: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    acc.append(sess.run(accuracy, feed_dict={x1: mnist.test.images, y_: mnist.test.labels}))

blt.plot(acc)
blt.xlabel("steps")
blt.ylabel("accuracy")
blt.show()

print("--- %s seconds ---" % (time.time() - start_time))
