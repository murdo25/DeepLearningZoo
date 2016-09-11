import numpy as np
import matplotlib.pyplot as plt
import pandas


class Perceptron:
	

	def __init__(self):
		
		data = pandas.read_csv( 'Fisher.csv' )
		m = data.as_matrix()
		labels = m[:,0]
		labels[ labels==2 ] = 1  # squash class 2 into class 1
		self.labels = np.atleast_2d( labels ).T
		self.features = m[:,1:5]
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










