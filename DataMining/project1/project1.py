import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import svm

'''
 File Name TEMP_FILE_NAME is used to store 
 the pickdataclass output and splitData2TestTrain as Input. 
''' 
TEMP_FILE_NAME = 'test.out' 
'''
 Convert the string to corresponding integers. 
	ASCII(<LETTER>) - 64
	64=ASCIIVALUE('A')-1
'''
def letter_2_digit_convert(mystr):
	mylist = []
	mystr = mystr.upper()
	for i in mystr:
		if i.isalpha():
			mylist.append(ord(i)-64)
	return mylist

'''
 Spliting the data based on the classids given by the 
 letter_2_digit_convert function.
 stroing the output into TEMP_FILE_NAME
'''

def pickDataClass(filename, class_ids):
	data = np.genfromtxt(filename, delimiter=',')
	listOfClassifierColumn = []
	for i in class_ids:
		a = np.where(data[0] == i) # returns index locations of the perticular class
		listOfClassifierColumn.extend(np.array(a).tolist()) # appending columns into a string 
	listOfClassifierColumn = [item for sublist in listOfClassifierColumn for item in sublist] # forming a array	
	np.savetxt(TEMP_FILE_NAME, data[:,listOfClassifierColumn], fmt="%i", delimiter=',')

'''
 splitData2TestTrain takes filename, number_per_class, test_instances
 split the data into testVector, testLabel, trainVector, trainLabel
 Get list of train instances, test instances, strip them and add into respective matrix.
'''

def splitData2TestTrain(filename, number_per_class, test_instances):
	start, end = test_instances.split(":")
	listTest  = list(range(int(start), int(end)+1))
	listTrain = list((set(list(range(0,number_per_class)))-set(listTest)))
	Training = []
	Test = []
	data = np.genfromtxt(filename, delimiter=',')	
	for i in xrange(0, data[0].size, number_per_class):
		templistTest=[x+i for x in listTest]
		templistTrain=[x+i for x in listTrain]
		templistTest.sort()
		templistTrain.sort()
		if len(Test) == 0:
			Test = data[:,templistTest]
		else:
			Test= np.concatenate((Test ,data[:,templistTest]), axis=1)
		if len(Training) == 0:
			Training = data[:,templistTrain]
		else:
			Training= np.concatenate((Training , data[:,templistTrain]), axis=1)
	
	return Test[1:,], Test[0], Training[1:,], Training[0]
'''

 Stores the np type array into fileName after stacking label over train. 

'''
def store(trainX, trainY, fileName):
	np.savetxt(fileName, np.vstack((trainY, trainX)), fmt="%i", delimiter=',')

'''
 printAccuracy returns the accuracy comparision from Ytest and calculated label

'''
SVM = svm.SVC()
def printAccuracy(sampleLabel, calculatedLabel):
	err_test_padding = sampleLabel - calculatedLabel
	TestingAccuracy_padding = (1-np.nonzero(err_test_padding)[0].size/float(len(err_test_padding)))*100
	return (TestingAccuracy_padding)
'''
 Linear regression:
	Xtest_padding : formed by adding ones to bottom of Xtest
	Xtrain_padding: formed by adding ones to bottom of Xtrain
    Ytrain_Indent : Form array with class label index as 1 other are zero.
		e.g LabelVector = [1,5]
		| 1 1 1 0 0 0 |
		| 0 0 0 1 1 1 | 
	return Accuracy. 
'''
def linear(Xtrain, Xtest,Ytrain,Ytest):
	RowToFill = 0
	A_train = np.ones((1,len(Xtrain[0])))
	A_test = np.ones((1,len(Xtest[0])))      
	Xtrain_padding 	= np.row_stack((Xtrain,A_train))
	Xtest_padding 	= np.row_stack((Xtest,A_test))
	element, count = np.unique(Ytrain,return_counts=True)
	Ytrain_Indent = np.zeros((len(element), count[0]*len(element)))
	for i in count:
		Ytrain_Indent[RowToFill,RowToFill*i:RowToFill*i+i] = np.ones(i)
		RowToFill+=1

	B_padding = np.dot(np.linalg.pinv(Xtrain_padding.T), Ytrain_Indent.T)
	Ytest_padding = np.dot(B_padding.T, Xtest_padding)
	Ytest_padding_argmax = np.argmax(Ytest_padding,axis=0)+1
	
	err_test_padding = Ytest - Ytest_padding_argmax
	TestingAccuracy_padding = (1-np.nonzero(err_test_padding)[0].size/float(len(err_test_padding)))*100
	return TestingAccuracy_padding
'''

 kNearestNeighbor(X_train, y_train, X_test, k)
 return y_test
 Find eucledean distance between points, shorlist least k
 returns the dominent label of k.
 
'''
def kNearestNeighbor(X_train, y_train, X_test, k):
	predictions = []
	for i in range(len(X_test[0])):
		distances = []
		targets = []
		tempTest = X_test[:,i]
		for iNr in range(len(X_train[0])):
			distance = np.sqrt(np.sum(np.square(tempTest - X_train[:,iNr])))
			distances.append([distance, iNr])
		distances = sorted(distances)
		for iNr in range(k):
			index = distances[iNr][1]
			targets.append(y_train[index])
		predictions.append(max(set(targets), key=targets.count))

	predictions = list(int(i) for i in predictions)
	return predictions
'''
 Using the Scikit library to implement the svm method.

'''

def svmClassifier(train, trainLabel, test, testLabel):
	train = train.transpose()
	test = test.transpose()
	SVM.fit(train,trainLabel)
	SVM.predict(test)
	return test
'''

 centroid method compares the eucledean distance between the 
 nearest centroid. 

'''

def centroid(trainVector, trainLabel, testVector, testLabel):
	jj = []
	result = []
	for j in xrange(0, len(trainVector[0]), 8):
		columnMean = []
		columnMean.append(trainLabel[j])
		for i in range(len(trainVector)):
			columnMean.append(np.mean(trainVector[i,j:j+7]))		
		if not len(jj):
			jj = np.vstack(columnMean)
		else:
			jj = np.hstack((jj,(np.vstack(columnMean))))

	for iN in range(len(testVector[0])):
		distances = []	
		for m in range(len(jj[0])):
			euclead = np.sqrt(np.sum(np.square(testVector[:,iN] - jj[1:, m])))
			distances.append([euclead,int(jj[0,m])])
			distances = sorted(distances, key=lambda distances: distances[0])
		result.append(distances[0][1])
	return result
'''
 General task function for the Task C and Task D.
'''
def TaskC(userString):
	l = letter_2_digit_convert(userString)
	pickDataClass('HandWrittenLetters.txt', l)
	svmAccList = []
	centroidAccList = []
	knnAccList = []
	linearAccList = []
	print ('Calculating'+'.'*50)
	for i in range(5,39,5):
		testVector, testLabel, trainVector, trainLabel = splitData2TestTrain('test.out', 39, str(i)+':38')
		centroidresult = centroid(trainVector, trainLabel, testVector, testLabel)
		centroidAccList.append(printAccuracy(testLabel, centroidresult))
	argX = ['','(5, 34)', '(10,29)',  '(15,24)' ,'(20,19)', '(25,24)' , '(30,9)' ,'(35,4)']
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.set_xticklabels(argX, minor=False)
	ax1.set_xlabel('(Train, Test)')
	ax1.set_ylabel('Accuracy (%)')
	ax1.plot(centroidAccList,'ro')
	ax1.plot(centroidAccList)
	ax1.set_title('Centroid Classification.')
	for i,j in zip(range(7),centroidAccList):
		ax1.annotate("%.2f"%j,xy=(i+0.2,j))
	plt.show()

def main(argv):
	sys.stderr.write("\x1b[2J\x1b[H")
	if argv[1].upper() == 'A':
		# Task A
		print('''\tTASK A :\n
				Use the data-handler to select "A,B,C,D,E" classes from the hand-written-letter data.
				From this smaller dataset, Generate a training and test data: for each class.
				using the first 30 images for training and the remaining 9 images for test.
				Do classification on the generated data using the four classifers.''')

		
		pickDataClass('HandWrittenLetters.txt', letter_2_digit_convert('ZYXWQ'))
		testVector, testLabel, trainVector, trainLabel = splitData2TestTrain(TEMP_FILE_NAME, 39, '30:38')
		svmMatrix = svmClassifier(trainVector, trainLabel, testVector, testLabel)
		centroidresult = centroid(trainVector, trainLabel, testVector, testLabel)
		linearresult = linear(trainVector, testVector, trainLabel, testLabel)
		knnresult = kNearestNeighbor(trainVector, trainLabel,testVector, 5)
		svmresult = SVM.score(svmMatrix,testLabel)
		svmresult *= 100 
		print ('\n\nAccuracy of SVM is %0.2f \n' %  svmresult)
		print ('Accuracy of Centroid  is %0.2f\n' % printAccuracy(testLabel,centroidresult))
		print ('Accuracy of Linear is %0.2f\n' % linearresult)
		print ('Accuracy of 5-NN is %0.2f\n' %printAccuracy(testLabel,knnresult))

	elif argv[1].upper() == 'B':

		# Task B
		print(''' \t TASK B : \n 
				On ATNT data, run 5-fold cross-validation (CV) using  each of the
				four classifiers: KNN, centroid, Linear Regression and SVM.
				If you don't know how to partition the data for CV, you can use the data-handler to do that.
				Report the classification accuracy on each classifier.
				Remember, each of the 5-fold CV gives one accuracy. You need to present all 5 accuracy numbers
				for each classifier. Also, the average of these 5 accuracy numbers.''')
		svmAccList = []
		centroidAccList = []
		knnAccList = []
		linearAccList = []
		print ('Calculating'+'.'*50)
		for i in range(0,10,2):
			testVector, testLabel, trainVector, trainLabel = splitData2TestTrain('ATNTFaceImages400.txt', 10, str(i)+':'+str(i+1))
			svmMatrix = svmClassifier(trainVector, trainLabel, testVector, testLabel)
			centroidresult = centroid(trainVector, trainLabel, testVector, testLabel)
			linearAccList.append(linear(trainVector, testVector, trainLabel, testLabel))
			knnresult = kNearestNeighbor(trainVector, trainLabel,testVector, 5)
			
			svmresult = SVM.score(svmMatrix,testLabel)
			svmresult *= 100 
			svmAccList.append(svmresult)
			centroidAccList.append(printAccuracy(testLabel, centroidresult))
			knnAccList.append(printAccuracy(testLabel, knnresult))
			
		print ('\nAverage accuracy of SVM after 5-Fold is %0.2f'%(sum(svmAccList)/len(svmAccList)))
		print (svmAccList)
		print ('\nAverage accuracy of Centroid after 5-Fold is %0.2f'%(sum(centroidAccList)/len(centroidAccList)))
		print (centroidAccList)
		print ('\nAverage accuracy of 5-NN after 5-Fold is %0.2f'%(sum(knnAccList)/len(knnAccList)))
		print (knnAccList)
		print ('\nAverage accuracy of Linear after 5-Fold is %0.2f'%(sum(linearAccList)/len(linearAccList)))
		print (linearAccList)

	elif argv[1].upper() == 'C':
		# Task C
		print(''' \t TASK C : \n 
				On handwritten letter data, fix on 10 classes. Use the data handler to generate training and test data files.
				Do this for seven different splits:  (train=5 test=34), (train=10 test=29),  (train=15 test=24) ,
				(train=20 test=19), (train=25 test=24) , (train=30 test=9) ,  (train=35 test=4). 
				On these seven different cases, run the centroid classifier to compute average test image classification
				accuracy. Plot these 7 average accuracy on one curve in a figure. What trend can you observe?
				When do this task, the training data and test data do not need be written into files.	''')
		TaskC('ABCDEFGHIJ')

	elif argv[1].upper()=='D':
		print(''' \t TASK D:\n
				Repeat task (D) for another different 10 classes.  You get another 7 average accuracy.
				Plot them on one curve in the same figure as in task (D). Do you see some trend?''')
		TaskC('KLMNOPQRST')
	else:
		print('Usage %s : A/B/C/D'%argv[0])
	sys.exit(2) 

if __name__ == '__main__':
	main(sys.argv)

