import numpy as np
import matplotlib.pyplot as plt
import pandas


def unpickle( file ):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


class Perceptron:
	

	def __init__(self):
		'''
		file = 'cifar-10-batches-py/data_batch_1'
		import cPickle
		fo = open(file, 'rb')
 		dict = cPickle.load(fo)
  		fo.close()
    		 
		data = dict
 		'''
 		data = unpickle( 'cifar-10-batches-py/data_batch_1' )
		self.features = data['data']
		#print(type(self.features))
		#print(self.features)
		
		
		self.labels = data['labels']
		self.labels = np.atleast_2d( self.labels ).T
 
		# squash classes 0-4 into class 0, and squash classes 5-9 into class 1
		self.labels[ self.labels < 5 ] = 0
		self.labels[ self.labels >= 5 ] = 1
		
		
		print(type(self.labels))
		print(self.labels)
		
		
		self.weights = np.random.randn(4)
		self.bias = 0.0
		self.learningRate = .01
		self.fig = []
		self.step = 0
		self.accuracy = 0
		

	def printer(self):
		print("\n\nWeights")
		print(self.weights)
		print("Bias")
		print(self.bias)
		print("All the labels")
		print(self.labels)
		print("All the Data")
		print(self.features)
		print("learning Rate")
		print(self.learningRate)
		
	
	
	
	def plotter(self,score,step):
			#remember %inline
			lst = []
			lst.append(step)
			lst.append(score)
			
			self.fig.append(score)



	def showPlot(self):
			print(self.fig)
			plt.plot(self.fig)
			plt.show()	
			


	def Threshold(self,Input):
		if ((np.dot(self.weights,Input) + self.bias) >= 0.0):
			return 1
		else:
			return 0	

	def perceptronLearningRule(self,target,inputVector):
		result = self.Threshold(inputVector)
		
		for i in range(len(inputVector)):
			self.weights[i] += (self.learningRate*(target-result)*inputVector[i])
		
		self.bias += self.learningRate*(target-result)
		
		if result == target:
			self.accuracy += 1	
					
		#self.plotter((result/target),self.step)
		#print(self.bias)
		




per = Perceptron()

#per.printer()

theAccuracy = []
theL2Norm = []

for runs in range(100):
	per.accuracy = 0
	for i in range(len(per.features)):
		per.perceptronLearningRule(per.labels[i],per.features[i])
	print(per.accuracy)
	print(len(per.features))
	theAccuracy.append(per.accuracy/len(per.features))
	
	theL2Norm.append(np.log(np.sqrt(np.dot(per.weights,per.weights))))
	
	#per.showPlot()

print(theAccuracy)
plt.plot(theAccuracy)
plt.show()

plt.plot(theL2Norm)
plt.show()










