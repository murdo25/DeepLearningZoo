import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

#
# ==================================================================
#

keep_prob = []
classification_accuracy = []
test_accuracy = []
training_accuracy = []
baseline_accuracy = []

itter = range(11)
itter.pop(0)

for i in itter:
    
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
    mask1 = tf.placeholder( tf.float32, shape=[None,500], name="mask1")
    mask2 = tf.placeholder( tf.float32, shape=[None,500], name="mask2")
    mask3 = tf.placeholder( tf.float32, shape=[None,1000], name="mask3")
    y_ = tf.placeholder( tf.float32, shape=[None, 10], name="y_" )
    x  = tf.placeholder( tf.float32, [None, 784], name="x" )
    
    
    W1 = weight_variable([784, 500])
    b1 = bias_variable([500])
    h1 = tf.nn.relu( tf.matmul( x, W1 ) + b1 )
    h1 *= mask1
    
    W2 = weight_variable([500, 500])
    b2 = bias_variable([500])
    h2 = tf.nn.relu( tf.matmul( h1, W2 ) + b2 )
    h2 *= mask2
    
    W3 = weight_variable([500, 1000])
    b3 = bias_variable([1000])
    h3 = tf.nn.relu( tf.matmul( h2, W3 ) + b3 )
    h3 *= mask3
    
    W4 = weight_variable([1000, 10])
    b4 = bias_variable([10])
    y_hat = tf.nn.softmax(tf.matmul(h3, W4) + b4)
    
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_hat), reduction_indices=[1]))
    xent_summary = tf.scalar_summary( 'xent', cross_entropy )
    
    correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_summary = tf.scalar_summary( 'accuracy', accuracy )
    
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
    
    #Drop prob 
    p = i * .1
    keep_prob.append(p)
    
    #Drop mask
    def Masks(size):
        msk = np.random.rand(1000,size) < p
        return msk
    
    for i in range( 150 ):
      
      #mask1 = Mask(500)
      _, acc = sess.run( [ train_step, accuracy ], 
               feed_dict={ x: images, y_: labels, mask1 : Masks(500), mask2 : Masks(500), mask3 : Masks(1000) } )
               
      #print( "step %d, training accuracy %g" % (i, acc) )
      if i == 149:
           training_accuracy.append(acc)
  
    #
    # ==================================================================
    #
    
    
    ones1 = np.ones((10000,500))  * p
    ones2 = np.ones((10000,500))  * p
    ones3 = np.ones((10000,1000)) * p
    
    
    final_acc = sess.run( accuracy, 
                feed_dict={ x: mnist.test.images, y_: mnist.test.labels, mask1: ones1, mask2: ones2, mask3: ones3 } )
    print( "test accuracy %g" % final_acc )
    test_accuracy.append(final_acc)
    
    
    
#SHOW
plt.figure(figsize=(17,10))
plt.grid(True)  

BASELINE = [final_acc] * len(keep_prob)
plt.plot(keep_prob,training_accuracy,label="Training")   
plt.plot(keep_prob,test_accuracy,label="Test")
plt.plot(keep_prob,BASELINE,label = "Baseline")


#base, = plt.plot(x, BASELINE, 'r--', label="Baseline")

plt.xlabel("Keep Probability")
plt.ylabel("Classification Accuracy")

plt.legend(bbox_to_anchor=(1, 0.2))

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


