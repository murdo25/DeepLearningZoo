
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
import tensorflow as tf
import numpy as np
import random 
 #
 # assumes list.txt is a list of filenames, formatted as
 #
 # ./lfw2//Aaron_Eckhart/Aaron_Eckhart_0001.jpg
 # ./lfw2//Aaron_Guiel/Aaron_Guiel_0001.jpg
 # ...
 #
  
#HYPER PARAMETERS:
margin = .9 
batch_size = 100
#batch_size = 10


#@Wingate
files = open( './list.txt' ).readlines()
 
data = np.zeros(( len(files), 250, 250, 1 ))
labels = np.zeros(( len(files), 1 ))
 
# a little hash map mapping subjects to IDs
ids = {}
scnt = 0
 
# load in all of our images
ind = 0
for fn in files:
 
    subject = fn.split('/')[3]
    if not ids.has_key( subject ):
        ids[ subject ] = scnt
        scnt += 1
    label = ids[ subject ] 
    data[ ind, :, :, :] = np.reshape(np.array( Image.open( fn.rstrip() ) ),(250,250,1))    
    labels[ ind ] = label
    ind += 1
 
# data is (13233, 250, 250)
# labels is (13233, 1)

sorted_data = {}


for i in range(len(labels)):
    #sort the faces into dictionary slots
    if(sorted_data.has_key(int(labels[i][0]))):
        sorted_data[int(labels[i][0])].append(data[i])
    else:
        sorted_data[int(labels[i][0])] = [data[i]]

def get_good_test_split(num = batch_size):
    good_test = []
    for i in range(num):
        indx = random.randint(len(sorted_data)-1000,len(sorted_data)-1)
        while(len(sorted_data[indx]) == 1):
            indx = random.randint(len(sorted_data)-1000,len(sorted_data)-1)
        good_test.append(indx) 
    return good_test

def get_bad_test_split(num = batch_size*2):
    test = []
    for i in range(num):
        indx = random.randint(len(sorted_data)-1000,len(sorted_data)-1)
        test.append(indx) 
    return test

def get_indx(num = batch_size):
    batch = []
    for i in range(num):
        #indx = random.randint(0,len(sorted_data)-1)
        indx = random.randint(0,len(sorted_data)-1000)
        while(len(sorted_data[indx]) == 1 ):
            indx = random.randint(0,len(sorted_data)-1000)       
        batch.append(indx)
    return batch        
 

def getBad_indx(num = batch_size*2):
    batch = []
    for i in range(num):
        #indx = random.randint(0,len(sorted_data)-1)
        indx = random.randint(0,len(sorted_data)-1000)
        while(len(sorted_data[indx]) > 1 ):
            indx = random.randint(0,len(sorted_data)-1000)       
        batch.append(indx)
    return batch        


def createTestSet(dataDict,num=batch_size):
    batch1 = np.zeros((((batch_size,250,250,1))))
    batch2 = np.zeros((((batch_size,250,250,1))))
    indx_list = get_good_test_split() 
    for i in range(num):
        #could mod to chose rand images not the first and second:
        batch1[i] = dataDict[indx_list[i]][0]   
        batch2[i] = dataDict[indx_list[i]][1]        
    return batch1, batch2


def createBadTestSet(dataDict,num=batch_size):
    batch1 = np.zeros((((batch_size,250,250,1))))
    batch2 = np.zeros((((batch_size,250,250,1))))
    indx_list = get_bad_test_split() 
    for i in range(num):
        #could mod to chose rand images not the first and second:
        batch1[i] = dataDict[indx_list[i]][0]   
        batch2[i] = dataDict[indx_list[batch_size+i]][0]        
    return batch1, batch2


def createBatch(dataDict,num=batch_size):
    batch1 = np.zeros((((batch_size,250,250,1))))
    batch2 = np.zeros((((batch_size,250,250,1))))
    indx_list = get_indx()
    for i in range(num):
        #could mod to chose rand images not the first and second:
        batch1[i] = dataDict[indx_list[i]][0]   
        batch2[i] = dataDict[indx_list[i]][1]        
    return batch1, batch2

def createBadBatch(dataDict,num = batch_size):
    batch1 = np.zeros((((batch_size,250,250,1))))
    batch2 = np.zeros((((batch_size,250,250,1))))
    indx_list = getBad_indx()
    for i in range(num):
        batch1[i] = dataDict[indx_list[i]][0]   
        batch2[i] = dataDict[indx_list[i+batch_size]][0]        
    return batch1, batch2

batch1, batch2 = createBatch(sorted_data)

batch3, batch4 = createBadBatch(sorted_data)



#-------------

def conv2d( in_var, filter_width, filter_height, name="conv2d",reuse=False ):
    output_dim = 64
    k_w = filter_width  # filter width/height
    k_h = filter_height
    d_h = 2  # x,y strides
    d_w = 2

    with tf.variable_scope( name,reuse=reuse ):
        W = tf.get_variable( "W", [k_h, k_w, in_var.get_shape()[-1], output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.02) )
        b = tf.get_variable( "b", [output_dim], initializer=tf.constant_initializer(0.0) )

        conv = tf.nn.conv2d( in_var, W, strides=[1, d_h, d_w, 1], padding='SAME' )
        #conv = tf.reshape( tf.nn.bias_add( conv, b ), conv.get_shape() )

    return conv

'''
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
'''

def weight_variable(shape):
  initial = tf.truncated_normal( shape, stddev=0.1 )
  return tf.Variable( initial )


def bias_variable(shape):
  initial = tf.constant( 0.1, shape=shape )
  return tf.Variable(initial)

def linear( in_var, output_size, name="linear", stddev=0.02, bias_val=0.0,reuse=False ):
    #print('invar') 
    #print(in_var.get_shape())
    shape = in_var.get_shape().as_list()
    #print("shape") 
    #print(shape[1])
    #print(type(shape[1]))
    #print('output')
    #print(type(output_size))
    with tf.variable_scope( name, reuse=reuse ):
        W = tf.get_variable( "W", [shape[1], output_size], tf.float32,
                             tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable( "b", [output_size],
                             initializer=tf.constant_initializer( bias_val ))

        return tf.matmul( in_var, W ) + b


def residualLayer(X1,name,reuse):
    conv1 = tf.nn.relu(conv2d(X1   ,3,3,reuse=reuse,name=name+"res_conv1"))
    conv2 = tf.nn.relu(conv2d(conv1,3,3,reuse=reuse,name=name+"res_conv2"))
    
    #print('X1')
    #print(X1.get_shape())
        
    conv2_pad = tf.pad(conv2,[[0,0],[24,23],[24,23],[0,0]])
    #print('conv2_pad')
    #print(conv2_pad.get_shape())
    
    return (conv2_pad + X1)
 
def siamese(x,name,reuse = False):
    #7x7conv,64,/2
    #print(x.get_shape())
    conv1 = conv2d(x,7,7,name= name + "first_conv",reuse=reuse)
    #print('conv1')
    #print(conv1.get_shape())
    
    #pool,/2
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    #print('pool1')
    #print(pool1.get_shape())
    #resnet 
    #   3x3conv,64
    #   3x3conv,64
    R1 = residualLayer(pool1,name + "res1",reuse=reuse)
    
    
    #resnet
    #   3x3conv,64
    #   3x3conv,64
    R2 = residualLayer(R1,name+"res2",reuse)
    
    
    #resnet
    #   3x3conv,64
    #   3x3conv,64
    R3 = residualLayer(R2,name+"res3",reuse)
    
    #avgpool or 2 by 2 conv   
    downSample = conv2d(R3,2,2,name=name+"downsample",reuse=reuse)
    #print('downsampled') 
    #print(downSample.get_shape())
    
    #flattened = tf.reshape(downSample, [batch_size * shapes[1] * shapes[2] * shapes[3]])
    
    flattened = tf.nn.avg_pool(downSample,ksize= [1, 2, 2, 1], strides =[1,2,2,1],padding='VALID')
    #print('flat')
    #print(flattened.get_shape())
    
    shapes = flattened.get_shape().as_list()
    flat = tf.reshape(flattened,(batch_size,shapes[1]*shapes[2]*shapes[3]))

    # fc[1000]
    fc1 = linear(flat,1000,name=name+"_linear",reuse=reuse)
    
    return fc1

def l1Norm(x1,x2):
    return tf.abs(x1 - x2)



# Declare computation graph

x1 = tf.placeholder( tf.float32, shape=[None, 250, 250, 1], name= "x1" )
x2 = tf.placeholder( tf.float32, shape=[None, 250, 250, 1], name= "x2" )
bad1 = tf.placeholder( tf.float32, shape=[None, 250, 250, 1], name='bad1')
bad2 = tf.placeholder( tf.float32, shape=[None, 250, 250, 1], name='bad1')


#--- Good
outputPair1 = siamese(x1,name ="syam",reuse = False)
outputPair2 = siamese(x2,name ="syam",reuse = True) 

#sig1 = tf.nn.sigmoid(outputPair1)
#sig2 = tf.nn.sigmoid(outputPair2)

#loss = tf.mul(.5,tf.square(l1Norm(sig1,sig2)))
loss = tf.mul(.5,tf.square(l1Norm(outputPair1,outputPair2)))




#for printing
acc = tf.reduce_mean(loss)

#Define Entropy
train_step1 = tf.train.AdamOptimizer(1e-3).minimize(loss)



#--- BAD 
outputPair3 = siamese(bad1,name ="syam",reuse=True) 
outputPair4 = siamese(bad2,name ="syam",reuse=True) 

sig3 = tf.nn.sigmoid(outputPair3)
sig4 = tf.nn.sigmoid(outputPair4)

norm = l1Norm(sig3,sig4)
#l1norm_diff = l1Norm(outputPair3,outputPair4)
#tryingsomething = tf.reduce_mean(l1norm_diff)

#     L = .5{max(0,m-X)}^2
loss2 = tf.mul(.5,tf.square(tf.maximum(0.0,tf.abs(tf.sub(margin, norm))))) 

acc2 = tf.reduce_mean(loss2)


#Define Entropy
train_step2 = tf.train.AdamOptimizer(1e-3).minimize(margin - acc2)









sess = tf.Session()
sess.run( tf.initialize_all_variables() )
summary_writer = tf.train.SummaryWriter('logs/', sess.graph)


simTrue = []
simFalse = []



for i in  tqdm.tqdm(xrange( 75 )):
     

     
    batch1, batch2 = createBatch(sorted_data)
    batch3, batch4 = createBadBatch(sorted_data)
    
    out = sess.run( [ train_step1, acc ], feed_dict={ x1: batch1, x2: batch2, bad1: batch3, bad2: batch4} )
    out2 = sess.run( [ train_step2, acc2 ], feed_dict={ x1: batch1, x2: batch2, bad1: batch3, bad2: batch4} )
   
    #extra step
    #batch1, batch2 = createBatch(sorted_data)
    #out = sess.run( [ train_step1, acc ], feed_dict={ x1: batch1, x2: batch2, bad1: batch3, bad2: batch4} )
    batch3, batch4 = createBadBatch(sorted_data)
    out2 = sess.run( [ train_step2, acc2 ], feed_dict={ x1: batch1, x2: batch2, bad1: batch3, bad2: batch4} )
    #print('out1', out[1])
    #print('out2', out2[1])
    
    #outputsim.append(out[1])
    #outputdif.append(out2[1])
    

    #print('accuracy', out[1])
    if (i % 5) == 0:
        t1,t2 = createTestSet(sorted_data)
        t3,t4 = createBadTestSet(sorted_data)
        TestOut1 = sess.run( [acc, acc2], feed_dict={x1:t3, x2:t4, bad1:t1, bad2:t2 })
        TestOut2 = sess.run( [acc, acc2], feed_dict={x1:t1, x2:t2, bad1:t3, bad2:t4 })
        print('###\n###\n###\n###') 
        print('same',TestOut2[0])
        print('Fake same',TestOut1[0])
        print('diff',TestOut2[1])
        print('Fake diff',TestOut1[1])
        
        simTrue.append(TestOut2[0])
        simFalse.append(TestOut1[0])


plt.figure(figsize=(17,10))
plt.grid(True)
plt.suptitle("Siamese Resnet", size=16)

sim,  = plt.plot(simTrue,label="sim")
dif, = plt.plot(simFalse,label="diff")

plt.legend()
plt.show()







