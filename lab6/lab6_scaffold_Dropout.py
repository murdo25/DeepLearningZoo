
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

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
mask = tf.placeholder( tf.float32, shape=[None,500], name ="mask")
train_time = tf.placeholder( tf.float32,shape =[None,1], name ="train time vs run time")
#train_time = tf.Variable(False)


W1 = weight_variable([784, 500])
b1 = bias_variable([500])
h1 = tf.matmul( x, W1 ) + b1
#tf.boolean_mask(h1,mask,name='boolean_mask')
h1 *= mask

W2 = weight_variable([500, 500])
b2 = bias_variable([500])
h2 = tf.matmul( h1, W2 ) + b2

W3 = weight_variable([500, 1000])
b3 = bias_variable([1000])
h3 = tf.matmul( h2, W3 ) + b3

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
train_time = True
sess.run( tf.initialize_all_variables() )

#
# ==================================================================
#

# NOTE: we're using a single, fixed batch of the first 1000 images
mnist = input_data.read_data_sets( "MNIST_data/", one_hot=True )

images = mnist.train.images[ 0:1000, : ]
labels = mnist.train.labels[ 0:1000, : ]

#dropout prob
p = 0.5

for i in range( 150 ):
  #make a dropout mask for this epoch
  mask1 = np.random.rand(1000,500) < p
  print("applying mask of size")
  print(mask1.shape)
  lst = mask1.tolist()
  training = 1
  
  _, acc = sess.run( [ train_step, accuracy ], feed_dict={ x: images, y_: labels, mask : lst,train_time : training  } )
  
  print( "step %d, training accuracy %g" % (i, acc) )
  
#  if i%10==0:
#      tmp = sess.run( accuracy, feed_dict={ x: mnist.test.images, y_: mnist.test.labels } )
#      print( "          test accuracy %g" % tmp )

#
# ==================================================================
#
 
 
#make nother dropout mask for this epoch
mask1 = np.random.rand(1000,500) < p
lst = mask1.tolist()
  
training = 0
  
_, acc = sess.run( [ train_step, accuracy ], feed_dict={ x: images, y_: labels, mask : lst,train_time : training  } )
print( "test accuracy %g" % final_acc )


