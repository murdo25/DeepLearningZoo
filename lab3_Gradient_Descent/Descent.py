##### create a linear score function
#use the log soft-max loss function
#To optimize the parameters use vanilla gradient descent
#%matplotlib inline
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
        
        #np.exp(vec)/(np.sum(np.exp(vec)))
        
        return (-np.log(newM))
    
    
    def numerical_gradient(self, loss1, loss2):
        return ((loss2 - loss1)/self.delta)
    
    
    def loss(self, scores):
        return np.sum(scores)/1000.0
        
        
    def compute_loss(self, features):
    	#dot product Weights and Features  
    	G.scores = np.dot(features,G.W)
    	#softmax output    
        MAX = G.softmax(G.scores)
        #retain only the values held by the correct label
        G.scores = MAX * oneHot
		#compute adv loss
        loss = G.loss(G.scores)
        
        return loss
        

oneHot = np.eye(10)[labels]    
G = GradientDescent(oneHot)
NUM_EPOCHS = 1000

#labels2 = labels[:,0]

#print(labels2.shape)


plt.figure()

Losses = []

for num in range(NUM_EPOCHS):
     
    loss1 = G.compute_loss(features)
   
   
    for i in range(np.shape(G.W)[0]):
    	for j in range(np.shape(G.W)[1]):
    	
	    	G.W[i][j] += G.delta
    
	    	loss2 = G.compute_loss(features)

	    	grad = G.numerical_gradient(loss1,loss2)
			
	        G.gradients[i][j] = grad

    		G.W[i][j] -= G.delta
    
    
    G.W = (G.W + (G.STEP_SIZE * -G.gradients))
    
    print(loss1)
    Losses.append(loss1)
    


plt.plot(Losses)
plt.show()

print("finished")    
