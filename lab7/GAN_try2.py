import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets( "MNIST_data/", one_hot=True )

#
# -------------------------------------------
#
# Global variables

batch_size = 128
z_dim = 10

#
# ==================================================================
# ==================================================================
# ==================================================================
#

def linear( in_var, output_size, name="linear", stddev=0.02, bias_val=0.0 ):
    shape = in_var.get_shape().as_list()

    with tf.variable_scope( name ):
        W = tf.get_variable( "W", [shape[1], output_size], tf.float32,
                              tf.random_normal_initializer( stddev=stddev ) )
        b = tf.get_variable( "b", [output_size],
                             initializer=tf.constant_initializer( bias_val ))

        return tf.matmul( in_var, W ) + b

def lrelu( x, leak=0.2, name="lrelu" ):
    return tf.maximum( x, leak*x )

def deconv2d( in_var, output_shape, name="deconv2d", stddev=0.02, bias_val=0.0 ):
    k_w = 5  # filter width/height
    k_h = 5
    d_h = 2  # x,y strides
    d_w = 2

    # [ height, width, in_channels, number of filters ]
    var_shape = [ k_w, k_h, output_shape[-1], in_var.get_shape()[-1] ]

    with tf.variable_scope( name ):    
        W = tf.get_variable( "W", var_shape,
                             initializer=tf.truncated_normal_initializer( stddev=0.02 ) )
        b = tf.get_variable( "b", [output_shape[-1]],
                             initializer=tf.constant_initializer( bias_val ))

        deconv = tf.nn.conv2d_transpose( in_var, W, output_shape=output_shape, strides=[1, d_h, d_w, 1] )
        deconv = tf.reshape( tf.nn.bias_add( deconv, b), deconv.get_shape() )
    
        return deconv

def conv2d( in_var, output_dim, name="conv2d" ):
    k_w = 5  # filter width/height
    k_h = 5
    d_h = 2  # x,y strides
    d_w = 2

    with tf.variable_scope( name ):
        W = tf.get_variable( "W", [k_h, k_w, in_var.get_shape()[-1], output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.02) )
        b = tf.get_variable( "b", [output_dim], initializer=tf.constant_initializer(0.0) )

        conv = tf.nn.conv2d( in_var, W, strides=[1, d_h, d_w, 1], padding='SAME' )
        conv = tf.reshape( tf.nn.bias_add( conv, b ), conv.get_shape() )

        return conv

#
# ==================================================================
# ==================================================================
# ==================================================================
#

# the generator should accept a (tensor of multiple) 'z' and return an image
# z will be [None,z_dim]
def gen_model(z_vec):
    
    H1 = linear(z_vec, output_size=6272, name="g_1_linear")
    H1 = lrelu(H1)
    H1 = tf.reshape(H1,[batch_size,7,7,128])
    
    D2 = deconv2d( H1, [batch_size,14,14,128], name="g_2_deconv2d")
    D2 = lrelu(D2)
    
    D3 = deconv2d( D2, [batch_size,28,28,1], name="g_3_deconv2d")
    D3 = tf.sigmoid(D3)
    
    #reshape D3 to be [batch_size,784] for compatibility with the discriminator.
    image = tf.reshape(D3,[batch_size,784]) 
    
    return image

# -------------------------------------------
    
# the discriminator should accept a (tensor of muliple) images and
# return a probability that the image is real
# imgs will be [None,784]
def disc_model(images):
    # imgs will be [None,784]
    images = tf.reshape( images, [ batch_size, 28, 28, 1 ] )
    
    C1 = conv2d( images, 32, name="d_1_conv2d")
    C1 = lrelu(C1)
    
    C2 = conv2d( C1, 64, name="d_2_conv2d")
    C2 = lrelu( C2)
    C2 = tf.reshape( C2, [ batch_size, -1 ] )
    
    L3 = linear( C2, 1023, name="d_3_linear" )
    L4 = linear( L3, 1, name="d_4_linear")
    
    verdict = tf.sigmoid(L4)
    
    return verdict

#
# ==================================================================
# ==================================================================
#

# Create your computation graph, cost function, and training steps here!

# Placeholders should be named 'z' and ''true_images'
true_images = tf.placeholder( tf.float32, shape = [None,784], name ="true_images"  )
z = tf.placeholder( tf.float32, shape =  [None,z_dim], name ="z")
sampled_zs = z

# Training ops should be named 'd_optim' and 'g_optim'
# The output of the generator should be named 'sample_images'


#START GRAPH
sample_images = gen_model(sampled_zs)


with tf.variable_scope("d_") as scope:
    probabilities_true = disc_model(true_images) 
    scope.reuse_variables()
    probabilities_sampled = disc_model(sample_images)


t_vars = tf.trainable_variables()

d_loss = tf.reduce_mean(tf.log(probabilities_true) + tf.log(1.0 - probabilities_sampled))
d_vars = [var for var in t_vars if 'd_' in var.name]
d_optim = tf.train.AdamOptimizer( 0.0002, beta1=0.5 ).minimize( -d_loss, var_list=d_vars )


g_loss = tf.reduce_mean(tf.log(probabilities_sampled))
g_vars = [var for var in t_vars if 'g_' in var.name]
g_optim = tf.train.AdamOptimizer( 0.0002, beta1=0.5 ).minimize( -g_loss, var_list=g_vars )

d_acc = (tf.reduce_mean(probabilities_true) + tf.reduce_mean(1.0 - probabilities_sampled))/2.0

#
# ==================================================================
# ==================================================================
# ==================================================================
#


sess = tf.Session()
sess.run( tf.initialize_all_variables() )
summary_writer = tf.train.SummaryWriter( "./tf_logs", graph=sess.graph )

def z_sample():
    return( np.random.uniform( -1, 1, size=(batch_size, z_dim) ).astype( np.float32 ))


for i in range( 5000 ):
    batch = mnist.train.next_batch( batch_size )
    batch_images = batch[0]
    
    sampled_zs = z_sample()
    sess.run( d_optim, feed_dict={ z:sampled_zs, true_images: batch_images } ) 

    for j in range(3):
        sampled_zs = z_sample()
        sess.run( g_optim, feed_dict={ z:sampled_zs } )
    
    if i%10==0:
        d_acc_val,d_loss_val,g_loss_val = sess.run( [d_acc,d_loss,g_loss],
                                                    feed_dict={ z:sampled_zs, true_images: batch_images } )
        print "%d\t%.2f %.2f %.2f" % ( i, d_loss_val, g_loss_val, d_acc_val )

summary_writer.close()

#
#  show some results
#
sampled_zs = z_sample()
simgs = sess.run( sample_images, feed_dict={ z:sampled_zs } )
simgs = simgs[0:64,:]

tiles = []
for i in range(0,8):
    tiles.append( np.reshape( simgs[i*8:(i+1)*8,:], [28*8,28] ) )
plt.imshow( np.hstack(tiles), interpolation='nearest', cmap=matplotlib.cm.gray )
plt.colorbar()
plt.show()
