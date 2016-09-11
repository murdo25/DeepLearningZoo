#create a linear score function
#use the log soft-max loss function
#To optimize the parameters use vanilla gradient descent

import numpy as np



#@wingate
#---
def unpickle( file ):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
 
data = unpickle( 'cifar-10-batches-py/data_batch_1' )
 
features = data['data']
labels = data['labels']
labels = np.atleast_2d( labels ).T
 
N = 1000
D = 10
 
# only keep N items
features = features[ 0:N, : ] 
labels = labels[ 0:N, : ]
 
# project down into a D-dimensional space
features = np.dot( features, np.random.randn( 3072, D) )
 
# whiten our data - zero mean and unit standard deviation
features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
#---


class GradientDescent:
    
    def __init__(self):
       
        
        self.delta = .00001
        self.W = np.random.rand(10,10)
        
        
    def softmax(self,vec):
        return (np.exp(vec)/np.sum(np.exp(vec),axis=0))    
        
    #def printr(self):
        #print("features ")
        #print(self.features)
        
        print("weights")
        print(self.W)
        
        
    
        
G = GradientDescent()



