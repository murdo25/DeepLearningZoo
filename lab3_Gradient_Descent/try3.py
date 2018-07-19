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
    	self.savedScores = np.zeros((10,1000))
    
    def softmax(self,vec,labels):
        #vec -= np.max(vec)
        newM = np.zeros((1000,10))
        
        for slot in range(1000):
                
                #vec[labels[column]] -= np.max(vec[column])
            newM = np.exp(vec[slot][labels[slot]])/np.sum(np.exp(vec[slot]))
        
        return (-np.log(newM))
    
    '''
    def softmax(self,vec):
        #vec -= np.max(vec)
        newM = np.zeros((1000,10))
        
        for column in range(1000):
                vec[column] -= np.max(vec[column])
                newM[column] = np.exp(vec[column])/np.sum(np.exp(vec[column]))
        
        return (-np.log(newM))
    '''
    
    def numerical_gradient(self, loss1, loss2):
        return ((loss2 - loss1)/self.delta)
    
    
    def loss(self, scores):
        #return np.sum(scores)/1000.0
        return np.mean(scores)
        
        
    def compute_loss(self, features,labels):
    	#dot product Weights and Features  
    	G.scores = np.dot(features,G.W)
        self.savedScores = G.scores

    	#softmax output    
        MAX = G.softmax(G.scores,labels)
        #retain only the values held by the correct label
        G.scores = MAX * oneHot
		#compute adv loss
        loss = G.loss(G.scores)
        
        return loss
     
    def accuracy(self,scores,labels):

        guesses = np.argmax(scores,axis=1)
        correct = np.equal(guesses,labels)

        return np.count_nonzero(correct)/1000.0
        
        
oneHot = np.eye(10)[labels]    
print("init")
G = GradientDescent(oneHot)
NUM_EPOCHS = 1000


Accuracys = []
Losses = []

for num in range(NUM_EPOCHS):
    print(num)
    loss1 = G.compute_loss(features,labels)
    
    Accuracys.append(G.accuracy(G.savedScores,labels))
    
    
    
    for i in range(np.shape(G.W)[0]):
    	for j in range(np.shape(G.W)[1]):
    	
	    	G.W[i][j] += G.delta
    		
	    	loss2 = G.compute_loss(features,labels)

	    	grad = G.numerical_gradient(loss1,loss2)
			
	        G.gradients[i][j] = grad

    		G.W[i][j] -= G.delta
    
    
    G.W = (G.W + (G.STEP_SIZE * -G.gradients))
    
    Losses.append(loss1)



plt.figure()
plt.ylabel('Softmax Cost')
plt.xlabel('epochs')
plt.plot(Losses)
plt.show()


#-----
plt.figure()
plt.ylabel('Classification Accuracy')
plt.xlabel('epochs')
plt.plot(Accuracys)
plt.show()

print("finished")    




