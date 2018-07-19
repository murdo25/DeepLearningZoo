import pickle
import random
import rospy
import time
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import numpy as np



with open('myLittlePickle_900.pkl','rb') as handle:
    data = pickle.load(handle)
    
train = []
test  = []
    
for i in range(len(data)):
    if(i % 10==0):
        test.append(data[i])
    else:
        train.append(data[i])

# Global variables
batch_size = 50


def getBatch(size):
    batch = []
    for i in range(size):
        irand = random.randint(0,len(train)-1)
        batch.append(train[irand])
    return batch
    
def getTest(size):
    batch = []
    for i in range(size):
        irand = random.randint(0,len(test)-1)
        batch.append(test[irand])
    return batch
    
           
def positions(batch,size):
    #finger = f
    #base = b
    startF =  np.zeros((size,3))
    startB =  np.zeros((size,3))
    endF =    np.zeros((size,3))
    endB =    np.zeros((size,3))
    servos = np.zeros((size,1))
    
    for i in range(size):
            #start Finger
            startF[i][0] = 100 * batch[i][0].position.x
            startF[i][1] = 100 * batch[i][0].position.y
            startF[i][2] = 100 * batch[i][0].position.z
            #start Base
            startB[i][0] = 100 * batch[i][1].position.x
            startB[i][1] = 100 * batch[i][1].position.y
            startB[i][2] = 100 * batch[i][1].position.z 
            #Servos
            servos[i]    =  batch[i][2]  
            #result Finger
            startF[i][0] = 100 * batch[i][3].position.x
            startF[i][1] = 100 * batch[i][3].position.y
            startF[i][2] = 100 * batch[i][3].position.z
            #result Base
            startB[i][0] = 100 * batch[i][4].position.x
            startB[i][1] = 100 * batch[i][4].position.y
            startB[i][2] = 100 * batch[i][4].position.z 
    return startF,startB,servos,endF,endB

#####################################################################          
            
            
def linear( in_var, output_size, name="linear", stddev=0.02, bias_val=0.0 ):
    shape = in_var.get_shape().as_list()

    with tf.variable_scope( name ):
        W = tf.get_variable( "W", [shape[1], output_size], tf.float32,
                              tf.random_normal_initializer( stddev=stddev ) )
        b = tf.get_variable( "b", [output_size],
                             initializer=tf.constant_initializer( bias_val ))

        return tf.matmul( in_var, W ) + b


######Define GRAPH     ##### 


#SF        = tf.placeholder( tf.float32, shape =  [None,3], name ="starting_finger_pos")
#SB        = tf.placeholder( tf.float32, shape =  [None,3], name ="starting_base_pos")
realServo = tf.placeholder( tf.float32, shape =  [None,1], name="real_servo_outputs")
#resultF   = tf.placeholder( tf.float32, shape =  [None,3], name ="starting_finger_pos")
#resultB   = tf.placeholder( tf.float32, shape =  [None,3], name ="starting_base_pos")
inputs   = tf.placeholder( tf.float32, shape =  [None,12], name ="concatenatedInputs")


W1 = tf.Variable(tf.random_normal([12,144],stddev=0.001),name='W1') # 144 = inputs**2
b1 = tf.Variable(tf.random_normal([144]))

W2 = tf.Variable(tf.random_normal([144,576],stddev=0.001),name='W1') # 1728 = inputs * 144
b2 = tf.Variable(tf.random_normal([576]))

W3 = tf.Variable(tf.random_normal([576,288],stddev=0.001),name='W2')
b3 = tf.Variable(tf.random_normal([288]))

W4 = tf.Variable(tf.random_normal([288,255],stddev=0.001),name='W2')
b4 = tf.Variable(tf.random_normal([255]))

W5 = tf.Variable(tf.random_normal([255,50],stddev=0.001),name='W2')
b5 = tf.Variable(tf.random_normal([50]))


#TOPOLOGY
#hidden layer 1

x2 = tf.nn.relu(tf.add(tf.matmul(inputs,W1), b1))


#fully connected hidden layer 2

x3 = tf.nn.relu(tf.add(tf.matmul(x2,W2),b2))
#shape = [batch_size, 1728]
#out_x3 = tf.nn.pool(x3,[1,2],"MAX",'VALID',name='POOLING')
#output shape = [batch_size, 846]
#x4 = tf.nn.relu(tf.add(tf.matmul(x3,W3),b3))
#x5 = tf.nn.relu(tf.add(tf.matmul(x4,W4),b4))
#x6 = tf.nn.relu(tf.add(tf.matmul(x5,W5),b5))


#servo_out = tf.reduce_mean(x6)
servo_out = tf.reduce_mean(x3)


loss = tf.reduce_mean(tf.abs( servo_out - realServo))# tf.floor(servo_out) - realServo))

##### OPTIMIZER #####
optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize( loss )


#init?
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
#sess = tf.Session()
#sess.run( tf.initialize_all_variables() )
losses = []

for i in range(1000):
    batch = getBatch(batch_size)     
    startF, startB, servo, endF, endB  = positions(batch,batch_size)
    #testnp = np.ones((50,50
    allInputs = np.concatenate((startF,startB,endF,endB),axis=1)
    #print(servo.shape) 
    _,l    = sess.run( [optim,loss], feed_dict={ realServo:servo, inputs:allInputs  } )
    
    #if i % 100 == 1: 
    #l  = sess.run( loss, feed_dict={ realServo:servo, inputs:allInputs  } )
    print('\n'+str(i)+ " , " + str(l))
    losses.append(l)
        #print(len(servo_out))
        #print(servo.shape)   
        #for k in range(15):      
        #    print(servo_out[0][k],servo[k])
    
    '''
    if i% 1000 == 1:
        tst = getTest(batch_size)
        tstartF, tstartB, tservo, tendF, tendB  = positions(tst,batch_size)
        allTestInputs = np.concatenate((tstartF,tstartB,tendF,tendB),axis=1)
        acc = sess.run( loss, feed_dict={realServo:tservo, inputs:allTestInputs } )
        print("\n"+str(i))
        #print(acc)
        
        #arr = x5.eval(session=sess)
        #print(arr) 
        
        for i in range(batch_size-1):
            print(acc[i],servo[i])
            pass    
    '''        
    #sess.run( d_optim, feed_dict={ SF:startF, SB:startB, realServo:servo, resultF:endF, resultB:endB  } )


#plt.figure()

    
#SHOW
plt.figure(figsize=(10,7))
plt.grid(True)  

plt.plot(losses,label="Test")
plt.xlabel("Steps, (in hundreds)")
plt.ylabel("Loss")

plt.legend(bbox_to_anchor=(1, 0.2))

plt.show()


''' 
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












