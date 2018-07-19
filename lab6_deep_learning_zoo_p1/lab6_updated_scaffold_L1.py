import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

trainAcc = []
testAcc = []


values = [0.1, 0.01, 0.001]
for j in range(len(values)):
    #
    # ==================================================================
    #
    
    def weight_variable(shape):
      initial = tf.truncated_normal( shape, stddev=0.1 )
      return tf.Variable( initial )
    
    def bias_variable(shape):
      initial = tf.constant( 0.1, shape=shape )
      return tf.Variable(initial)
    
    #
    # ==================================================================
    #
    
    # Declare computation graph
    
    y_ = tf.placeholder( tf.float32, shape=[None, 10], name="y_" )
    x = tf.placeholder( tf.float32, [None, 784], name="x" )
    lam = tf.placeholder( tf.float32, name="lam")
    
    
    
    W1 = weight_variable([784, 500])
    b1 = bias_variable([500])
    h1 = tf.nn.relu( tf.matmul( x, W1 ) + b1 )
    
    W2 = weight_variable([500, 500])
    b2 = bias_variable([500])
    h2 = tf.nn.relu( tf.matmul( h1, W2 ) + b2 )
    
    W3 = weight_variable([500, 1000])
    b3 = bias_variable([1000])
    h3 = tf.nn.relu( tf.matmul( h2, W3 ) + b3 )
    
    W4 = weight_variable([1000, 10])
    b4 = bias_variable([10])
    y_hat = tf.nn.softmax(tf.matmul(h3, W4) + b4)
    
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_hat), reduction_indices=[1]))
    xent_summary = tf.scalar_summary( 'xent', cross_entropy )
    
    correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_summary = tf.scalar_summary( 'accuracy', accuracy )
    
    regularizer = ( tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(b1)) +
                    tf.reduce_sum(tf.abs(W2)) + tf.reduce_sum(tf.abs(b2)) +
                    tf.reduce_sum(tf.abs(W3)) + tf.reduce_sum(tf.abs(b3)) +
                    tf.reduce_sum(tf.abs(W4)) + tf.reduce_sum(tf.abs(b4)) )
                    
    
    cross_entropy = cross_entropy + lam * regularizer
    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    #
    # ==================================================================
    #
    
    sess = tf.Session()
    sess.run( tf.initialize_all_variables() )
    
    #
    # ==================================================================
    #
    
    # NOTE: we're using a single, fixed batch of the first 1000 images
    mnist = input_data.read_data_sets( "MNIST_data/", one_hot=True )
    
    images = mnist.train.images[ 0:1000, : ]
    labels = mnist.train.labels[ 0:1000, : ]
    
    
    
    
    for i in range( 150 ):
      lam_ = values[j]
      
      
      _, acc = sess.run( [ train_step, accuracy ], feed_dict={ x: images, y_: labels, lam: lam_ } )
      print( "step %d, training accuracy %g" % (i, acc) )
      if i == 149:
            trainAcc.append(acc) 
    #  if i%10==0:
    #      tmp = sess.run( accuracy, feed_dict={ x: mnist.test.images, y_: mnist.test.labels } )
    #      print( "          test accuracy %g" % tmp )
    
    #
    # ==================================================================
    #
    
    final_acc = sess.run( accuracy, feed_dict={ x: mnist.test.images, y_: mnist.test.labels } )
    print( "test accuracy %g" % final_acc )
    testAcc.append(final_acc)
    
    
    
#init
plt.figure()
plt.grid(True)  


trainAcc_, = plt.plot(values,trainAcc,label="Training")   
testAcc_, = plt.plot(values,testAcc,label="Test")

plt.xlabel("Keep Probability")
plt.ylabel("Classification Accuracy")
#plt.legend(handles=[trainAcc, testAcc], loc=4)
plt.legend()

plt.show()
    
'''    
def graphData(x, trainingAcc, testAcc, Title): 
    plt.figure(figsize=(17,10))
    plt.grid(True)
    
    # Baseline was found earlier and is now a constant
    BASELINE = [0.8132] * len(x)
    base, = plt.plot(x, BASELINE, 'r--', label="Baseline")
    
    trainAcc, = plt.plot(x, trainingAcc, label="Training")
    tstAcc, = plt.plot(x, testAcc, label="Test")
    plt.xlabel("Keep Probability")
    plt.ylabel("Classification Accuracy")
    plt.title(Title, fontsize=17)
    plt.legend(handles=[trainAcc, tstAcc, base], loc=4)    

'''
