import numpy as np
import matplotlib.pyplot as plt


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

labels = (labels.flatten())

class GradientDescent:
    
    def __init__(self,oneHotMatrix):        
        self.delta = 0.000001
        self.W = np.random.rand(10,10)
        #@wingate
        
        self.STEP_SIZE = .1
        self.scores = np.zeros((10,1000))
        self.oneHot = oneHotMatrix 
        self.gradients = np.zeros((10,10))
    
    def softmax(self,vec):
        #vec -= np.max(vec)
        newM = np.zeros((1000,10))
        for column in range(1000):
                vec[column] -= np.max(vec[column])
                newM[column] = np.exp(vec[column])/np.sum(np.exp(vec[column]))
        return (-np.log(newM))
    
    def numerical_gradient(self, loss1, loss2):
        return (-((self.delta + loss2) - loss1)/self.delta)
    
    def loss(self, scores):
        return np.sum(scores)/1000.0
        
   
        
        
        
oneHot = np.eye(10)[labels]    
G = GradientDescent(oneHot)
NUM_EPOCHS = 100


plt.figure()

Losses = []

for num in range(NUM_EPOCHS):
    #dot product Weights and Features  
    G.scores = np.dot(features,G.W)
    MAX = G.softmax(G.scores)

	#retain only the values held by the correct label 
    G.scores = MAX * oneHot
       
    loss1 = G.loss(G.scores)
    G.W += G.delta

    G.scores = np.dot(features,G.W)
    
    MAX = G.softmax(G.scores)
    G.scores = MAX * oneHot
  
    loss2 = G.loss(G.scores)
	

    grad = G.numerical_gradient(loss1,loss2)
    G.gradients.fill(grad)
    G.W -= G.delta
    

    G.W = (G.W * G.STEP_SIZE * -G.gradients)





