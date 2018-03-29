from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np
import sys

def pickDataClass(filename, class_ids):
	data = np.genfromtxt(filename, delimiter=',')
	listOfClassifierColumn = []
	for i in class_ids:
		a = np.where(data[0] == i) # returns index locations of the perticular class
		listOfClassifierColumn.extend(np.array(a).tolist()) # appending columns into a string 
	listOfClassifierColumn = [item for sublist in listOfClassifierColumn for item in sublist] # forming a array	
	np.savetxt('test.out', data[:,listOfClassifierColumn], fmt="%i", delimiter=',')

def calcHungarian(yTrue, yPred):
	C = confusion_matrix(yTrue, yPred)
	print('Confusion Matrix')
	print(C) # Task of printing confusion matrix
	C = C.T
	ind = linear_assignment(-C)
	C_opt = C[:,ind[:,1]]
	print('Reordered Matrix')
	print(C_opt) # Task of printing reordered confusion matrix
	acc_opt = np.trace(C_opt)/float(np.sum(C_opt))
	print('Accuracy = %.2f'%(acc_opt*100)+'%') # Calculating the Accuracy using reordered matrix

def main(argv):
	if argv[1].upper()=='A':
		print('*'*50)
		print('TASK A: For  k-means on AT&T 100 images, set K=10. \n Obtain confusion matrix. \n Re-order the confusion matrix and \n obtain accuracy.')
		print('*'*50)

		pickDataClass('ATNTFaceImages400.txt', range(1,11))
		data = np.genfromtxt('test.out', delimiter=',')
		kmeans = KMeans(n_clusters=10, random_state=0).fit(data[1:,].T) # data with 10 class (without label)
		calcHungarian(data[0],kmeans.labels_+1)
	elif argv[1].upper()=='B':
		print('*'*50)
		print('TASK B: For  k-means on AT&T 400 images, set K=40. \n Obtain confusion matrix. \n Re-order the confusion matrix and \n obtain accuracy.')
		print('*'*50)

		data = np.genfromtxt('ATNTFaceImages400.txt', delimiter=',')
		kmeans = KMeans(n_clusters=40, random_state=0).fit(data[1:,].T) # Complete data(without label)  
		calcHungarian(data[0],kmeans.labels_+1)
	elif argv[1].upper()=='C': 
		print('*'*50)
		print('TASK B: For  k-means on HandWrittenLetters, set K=26. \n Obtain confusion matrix. \n Re-order the confusion matrix and \n obtain accuracy.')
		print('*'*50)
		data = np.genfromtxt('HandWrittenLetters.txt', delimiter=',')
		kmeans = KMeans(n_clusters=26, random_state=0).fit(data[1:,].T) # Complete data(without label)  
		calcHungarian(data[0],kmeans.labels_+1)
	else:
		pickDataClass('ATNTFaceImages400.txt', [1,2])
		data = np.genfromtxt('test.out', delimiter=',')
		print(data[0])
		kmeans = KMeans(n_clusters=2, random_state=0).fit(data[1:,].T) # Complete data(without label) 
		print(kmeans.labels_) 
		calcHungarian(data[0],kmeans.labels_+1)
		print('Usage : python project2.py <A,B or C>')	
if __name__ == '__main__':
	main(sys.argv)

