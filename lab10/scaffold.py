'''
x1.Run the VGG network on the content and style images. Save the activations.
X2.Construct a content loss function, based on the paper
X3.Construct a style loss function, based on the paper
    Xa.For each layer specified in the paper (also noted in the code), you'll need to construct a Gram matrix
    Xb.That Gram matrix should match an equivalent Gram matrix computed on the style activations
X4.Construct an Adam optimizer, step size 0.1
5.Initialize all of your variables and reload your VGG weights
6.Initialize your optimization image to be the content image (or another image of your choosing)
7.Optimize!
'''

import time
import numpy as np
import tensorflow as tf
import vggNet as vgg16
from scipy.misc import imread, imresize, imshow
import matplotlib.pyplot as plt
#from PIL import Image
from matplotlib.pyplot import plot, ion, show
'''
alpha = 0.001
beta  = 1.0
'''
#alpha Content
alpha = 1     
#beta Style 
beta  = 10000


sess = tf.Session()

opt_img = tf.Variable( tf.truncated_normal( [1,224,224,3],
                                        dtype=tf.float32,
                                        stddev=1e-1), name='opt_img' )

tmp_img = tf.clip_by_value( opt_img, 0.0, 255.0 )

vgg = vgg16.vgg16( tmp_img, 'vgg16_weights.npz', sess )

style_img = imread( 'JPEG-Promo-15.jpg', mode='RGB' )
style_img = imresize( style_img, (224, 224) )
style_img = np.reshape( style_img, [1,224,224,3] )

content_img = imread( 'img7.jpg', mode='RGB' )
content_img = imresize( content_img, (224, 224) )
content_img = np.reshape( content_img, [1,224,224,3] )

layers = [ 'conv1_1', 'conv1_2',
           'conv2_1', 'conv2_2',
           'conv3_1', 'conv3_2', 'conv3_3',
           'conv4_1', 'conv4_2', 'conv4_3',
           'conv5_1', 'conv5_2', 'conv5_3' ]

ops = [ getattr( vgg, x ) for x in layers ]

content_acts = sess.run( ops, feed_dict={vgg.imgs: content_img } )
style_acts = sess.run( ops, feed_dict={vgg.imgs: style_img} )

#
# --- construct your cost function here
#

#content_loss     1/2 sum(F-P)^2
#where f is the feature map
#   and P is the 
#conten_loss = tf.mul(.5, tf.reduce_sum( tf.square(tf.subtract(vgg.conv4_2, content_acts[8]) ) ) ) 
#content_loss = tf.nn.l2_loss(tf.subtract(vgg.conv4_2, content_acts[8]))
content_loss = tf.nn.l2_loss(vgg.conv4_2 - content_acts[8])

#Style Loss is a Gram matrix:
#style_loss  = tf.reduce_sum(F(ik)*F(jk),axis=k)
gram_style = [style_acts[0],style_acts[2],style_acts[4],style_acts[7],style_acts[10]]
gram_vgg = [vgg.conv1_1, vgg.conv2_1, vgg.conv3_1, vgg.conv4_1, vgg.conv5_1]

output_grams = []

for i in range(len(gram_style)):
    #print(i)
    shapes = gram_style[i].shape[1]**2,gram_style[i].shape[3]
    #print(shapes)
    reshaped_s = gram_style[i].reshape(shapes)
    #print(reshaped_s.shape)
    s_gram = tf.matmul(reshaped_s,reshaped_s,transpose_a=True)
    #print(s_gram.get_shape())
    #print(s_gram)
    reshaped_vgg = tf.reshape(gram_vgg[i],shapes)
    vgg_gram = tf.matmul(reshaped_vgg,reshaped_vgg,transpose_a=True)
    output_grams.append(tf.nn.l2_loss(vgg_gram-s_gram))
    
    
style_sum = tf.reduce_sum(output_grams)
#style_sum = tf.multiply(style_sum, (1.0/822083584.0))
style_sum = style_sum * (1.0/822083584.0)
style_sum = style_sum/5.0
#print(style_sum.get_shape())

al_content = alpha * content_loss
bta_style  = beta * style_sum

#loss = content_loss
#loss = tf.add( alpha * content_loss, beta * style_sum )  
loss = tf.add(al_content,bta_style)
    
# Relevant snippets from the paper:
#   For the images shown in Fig 2 we matched the content representation on layer 'conv4_2'
#   and the style representations on layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1' and 'conv5_1'
#   The ratio alpha/beta was  1x10-3
#   The factor w_l was always equal to one divided by the number of active layers (ie, 1/5)

# --- place your adam optimizer call here
#     (don't forget to optimize only the opt_img variable)
optim = tf.train.AdamOptimizer(learning_rate=0.1).minimize( loss,var_list=[opt_img] )




# this clobbers all VGG variables, but we need it to initialize the
# adam stuff, so we reload all of the weights...
sess.run( tf.initialize_all_variables() )
vgg.load_weights( 'vgg16_weights.npz', sess )

# initialize with the content image
print(sess.run([loss, opt_img.assign( content_img )]))


lst = []

# --- place your optimization loop here
for i in range(400):
    output = sess.run([loss,optim])
    #print(i,output[0],output[2],output[3])
    #lst.append(output[0])

stuffs = sess.run(opt_img)
imshow(stuffs[0])





