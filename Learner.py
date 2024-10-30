import Network
import math
import pandas as pd
import numpy as np
import ClassificationInfo
from ClassificationInfo import Accuracy
from sklearn.preprocessing import OneHotEncoder

class Learner: 

    def __init__ (self, data, classificationType, classPlace):
        self.data = data
        self.classificationInfos = []
        self.classPlace = classPlace
        self.classificationType = classificationType
        if( classificationType == "classification"):
            self.folds  = self.crossValidateClassification(data, classPlace)
        elif( classificationType == "regression"):
            self.folds = self.crossValidateRegression(data, classPlace)
            self.accuracyThreshold = 0.05 * data[classPlace].mean()
        else:
            raise ValueError("Invalid classification type")
        self.learningRate = 0.01
        self.momentum = 0.9
        self.hiddenLayers = 2
        self.neuronsPerLayer = 5
        self.batchSize = 4
        self.features = self.data.shape[1] - 1
        self.classes = self.data[classPlace].unique()
        self.network = self.tuneData()
    
    def setNetwork(self, network): # remove later after testing
        self.network = network
    
    def setTestClass(self, testClass):
        self.testClass = testClass

    def tuneData(self):
        pass

    def crossValidateClassification(self, cleanDataset, classColumn ,printSteps = False):
         # 10-fold cross validation with stratification of classes
        if printSteps == True:
            print("Running cross calidation with stratification...")
        dataChunks = [None] * 10
        classes = np.unique(cleanDataset[classColumn])
        print("Classes: ")
        print(classes)
        print()
        dataByClass = dict()

        for uniqueVal in classes:
        # Subset data based on unique class values
            print("looking at class")
            print(uniqueVal)
            print()
            classSubset = cleanDataset[cleanDataset[classColumn] == uniqueVal]
            print("Class Subset: ")
            print(classSubset)
            print()
            print("Creating a subset of data for class " + str(uniqueVal) + " with size of " + str(classSubset.shape[0]))
            dataByClass[uniqueVal] = classSubset

            numRows = math.floor(classSubset.shape[0] / 10) # of class instances per fold
            print("Number of rows per fold: " + str(numRows))

            for i in range(9):
                classChunk = classSubset.sample(n=numRows)
                if printSteps:
                    print("Number of values for class " + str(uniqueVal), " in fold " + str(i+1) + " is: " + str(classChunk.shape[0]))
                if dataChunks[i] is None:
                    dataChunks[i] = classChunk
                else:
                    dataChunks[i] = pd.concat([dataChunks[i], classChunk])

                classSubset = classSubset.drop(classChunk.index)

        # the last chunk might be slightly different size if dataset size is not divisible by 10
            if printSteps == True:
                print("Number of values for class " + str(uniqueVal), " in fold " + str(10) + " is: " + str(classSubset.shape[0]))
            dataChunks[9] = pd.concat([dataChunks[9], classSubset])

        if printSteps == True:
            for i in range(len(dataChunks)):
                print("Size of fold " + str(i+1) + " is " + str(dataChunks[i].shape[0]))

        return dataChunks

    def crossValidateRegression(self, data, targetColumn, printSteps = False):
        if printSteps == True:
            print("Running cross validation with stratification...")
        dataChunks = [None] * 10
        binLabels, binEdges = pd.qcut(data[targetColumn], q=3, retbins=True, labels=False)
        uniqueBins = np.unique(binLabels)
        for binLabel in uniqueBins:
            print("Creating a subset of data for bin " + str(binLabel))
            binSubset = data[binLabels == binLabel]
            print("Bin Subset: ")
            print(binSubset)
            print()
            numRows = math.floor(binSubset.shape[0] / 10)
            for i in range(9):
                binChunk = binSubset.sample(n=numRows)
                if dataChunks[i] is None:
                    dataChunks[i] = binChunk
                else:
                    dataChunks[i] = pd.concat([dataChunks[i], binChunk])
                binSubset = binSubset.drop(binChunk.index)
            dataChunks[9] = pd.concat([dataChunks[9], binSubset])
        return dataChunks
    
    def train(self):
        batches = self.network.createBatches(self.data)
        batchIndex = 0 
        converged = False
        for i in range(1):
            print("BATCH")
            batch = batches[batchIndex % len(batches)]
            print(batch)
            print()
            testClasses = batch[self.classPlace].to_numpy()
            testData = batch.drop(columns=[self.classPlace])
            output = self.forwardPass(testData)
            print("OUTPUT: ")
            print(output)
            print()
            if(self.classificationType == "classification"):
                output = output[1]
                oneHot = np.zeros((len(testClasses), len(self.classes)))
                classesList = self.classes.tolist()
                for i in range(len(testClasses)):
                    oneHot[i][classesList.index(testClasses[i])] = 1
                self.backwardPass(oneHot)
            else:
                self.backwardPass(testClasses)
        #until convergence:
            # 
            #output = self.forwardpass()
            #self.backwardPass(output)
        pass

    def test(self, testData):
        classifications = ClassificationInfo.ClassificationInfo()
        testClasses = testData[self.classPlace].to_numpy()
        testData = testData.drop(columns=[self.classPlace])
        output = self.forwardPass(testData)
        if(self.classificationType == "classification"):
            output = output[0]
            for i in range(len(output)):
                print("Predicted: " + str(output[i]) + " Actual: " + str(testClasses[i]))
                classifications.addTrueClass([testClasses[i], output[i]])
                classifications.addConfusion(self.classificationAccuracy(testClasses[i], output[i]))
        elif(self.classificationType == "regression"):
            for i in range(len(output)):
                print("Predicted: " + str(output[i]) + " Actual: " + str(testClasses[i]))
                classifications.addTrueClass([testClasses[i], output[i]])
                classifications.addConfusion(self.regressionAccuracy(testClasses[i], output[i]))
        self.classificationInfos.append(classifications)


    def classificationAccuracy(self, trueClass, predClass):
        if trueClass == predClass:
            if trueClass == self.classes[0]:
                return Accuracy.TP
            else:
                return Accuracy.TN
        else:
            if trueClass == self.classes[0]:
                return Accuracy.FN
            else:
                return Accuracy.FP
        
    def regressionAccuracy(self, trueClass, predClass):
        if abs(trueClass - predClass) <= self.accuracyThreshold:
            if(trueClass < predClass):
                return Accuracy.TN
            else:
                return Accuracy.TP
        else:
            if(trueClass < predClass):
                return Accuracy.FN
            else:
                return Accuracy.FP
    
    def forwardPass(self, batch):
        print("FORWARD PASS")
        return self.network.forwardPass(batch)

    def backwardPass(self, testClasses):
        test = self.network
        print("BACKWARD PASS TESTING: ")
        test.printNetwork()

    def gradientDescent(self):
        pass
    
    def run(self):
        for fold in self.folds:
            trainData = self.data.drop(fold.index)
            testData = fold
            self.train(trainData)
            self.test(testData)

        return self.classificationInfos