import Network
import math
import pandas as pd
import numpy as np
import ClassificationInfo
from ClassificationInfo import Accuracy

class Learner: 

    def __init__ (self, data, classificationType, classPlace):
        self.testData = data.sample(frac=0.1)
        self.data = data.drop(self.testData.index)
        self.losses = []
        self.classPlace = classPlace
        self.errors = []
        self.classificationType = classificationType
        if( classificationType == "classification"):
            self.folds  = self.crossValidateClassification(self.data, classPlace)
        elif( classificationType == "regression"):
            self.folds = self.crossValidateRegression(self.data, classPlace)
            self.accuracyThreshold = 0.05 * data[classPlace].mean()
        else:
            raise ValueError("Invalid classification type")
        self.learningRate = 0.0001
        self.momentum = 0.9
        self.hiddenLayers = 1
        self.neuronsPerLayer = self.data.shape[0]
        self.batchSize = 10
        self.features = self.data.shape[1] - 1
        self.classes = self.data[classPlace].unique()
        self.network = self.tuneData()

    def getFolds(self):
        return self.folds
    
    def setTestClass(self, testClass):
        self.testClass = testClass
    
    def checkOscillation(self, losses):
        oscillations = 0 
        for i in range(len(losses)):
            loss = losses[i]
            if(i == (len(losses) - 1)):
                break
            if(i != 0):
                if np.any(losses[i] > losses[i - 1]):
                    oscillations += 1
        if oscillations > 5:
            return True
        else:
            return False
        
    def tuneData(self):
        self.momentum = 0.9
        self.learningRate = 0.1
        self.network = Network.Network(self.hiddenLayers, self.neuronsPerLayer, self.features, len(self.classes), self.classificationType, self.batchSize, self.classes)
        momentumSet = False
        learningRateSet = False
        foldIndex = 0
        print("TRAINING MOMENTUM")
        while not momentumSet:
            print("training momentum", str(self.momentum))
            fold = self.folds[foldIndex % len(self.folds)]
            if self.momentum <= 0.5:
                momentumSet = True
                self.momentum = 0.5
                break
            self.train(self.data.drop(fold.index))
            if self.checkOscillation(self.losses):
                self.momentum -= 0.1
            else:
                momentumSet = True
            foldIndex += 1
            
        print("TRAINING LEARNING RATE")

        while not learningRateSet:
            if self.learningRate == 0.0001:
                learningRateSet = True
                break
            self.train(self.data.drop(self.folds[foldIndex % len(self.folds)].index))
            if self.checkOscillation(self.losses):
                self.learningRate = self.learningRate / 10
            else:
                learningRateSet = True
            foldIndex += 1
        
        nueronsPerLayer = self.data.shape[0]
        nueronValues = np.linspace(20, nueronsPerLayer, 5).astype(int)
        accuracy = 0
        bestNuerons = 0
        print("TRAINING NEURONS PER LAYER")
        for nuerons in nueronValues:
            self.neuronsPerLayer = nuerons
            self.network = Network.Network(self.hiddenLayers, self.neuronsPerLayer, self.features, len(self.classes), self.classificationType, self.batchSize, self.classes)
            self.train(self.data.drop(self.folds[foldIndex % len(self.folds)].index))
            output = self.test(self.testData)
            print("OUTPUT")
            print(output)
            foldAccuracy = (output.getTP() + output.getTN()) / (output.getTP() + output.getTN() + output.getFP() + output.getFN())
            if(foldAccuracy > accuracy):
                accuracy = foldAccuracy
                bestNuerons = nuerons
            foldIndex += 1
        self.neuronsPerLayer = bestNuerons
        batchSizes = np.linspace(10, 100, 5).astype(int)
        print("TRAINING BATCH SIZE")
        accuracy = 0
        bestBatchSize = 0
        for batchSize in batchSizes:
            self.batchSize = batchSize
            self.network = Network.Network(self.hiddenLayers, self.neuronsPerLayer, self.features, len(self.classes), self.classificationType, self.batchSize, self.classes)
            self.train(self.data.drop(self.folds[foldIndex % len(self.folds)].index))
            output = self.test(self.testData)
            foldAccuracy = (output.getTP() + output.getTN()) / (output.getTP() + output.getTN() + output.getFP() + output.getFN())
            if(foldAccuracy > accuracy):
                accuracy = foldAccuracy
                bestBatchSize = batchSize
            foldIndex += 1
        self.batchSize = bestBatchSize
        print("BEST MOMENTUM:")
        print(str(self.momentum))
        print()

        print("BEST LEARNING RATE:")
        print(str(self.learningRate))
        print()
        
        print("BEST NEURONS PER LAYER:")
        print(str(self.neuronsPerLayer))
        print()

        print("BEST BATCH SIZE:")
        print(str(self.batchSize))
        print()

        print("Hidden Layers: ", str(self.hiddenLayers))
        print("Neurons Per Layer: ", str(self.neuronsPerLayer))
        print("Features: ", str(self.features))
        print("Classes: ", str(len(self.classes)))
        print("Classification Type: ", str(self.classificationType))
        print("Batch Size: ", str(self.batchSize))
        print("Classes: ", str(self.classes))
        print()
        return Network.Network(self.hiddenLayers, self.neuronsPerLayer, self.features, len(self.classes), self.classificationType, self.batchSize, self.classes)

    def getNetwork(self):
        return self.network

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
        binLabels, binEdges = pd.qcut(data[targetColumn], q=10, retbins=True, labels=False, duplicates='drop')
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
    
    def train(self, trainData):
        self.losses = []
        if self.network.getBatchSize() != self.batchSize:
            self.network.setBatchSize(self.batchSize)
        batches = self.network.createBatches(trainData)
        batchIndex = 0 
        converged = False
        #while self.network.checkConvergence() == False: --> USE ONCE CONVERGENCE FULLY IMPLEMENTED
        for i in range(10):
            #print("BATCH")
            batch = batches[batchIndex % len(batches)]
            #print(batch)
            #print()
            testClasses = batch[self.classPlace].to_numpy()
            testData = batch.drop(columns=[self.classPlace])
            output = self.forwardPass(testData)
            #print("OUTPUT: ")
            #print(output)
            #print()
            if(self.classificationType == "classification"):
                output = output[1]
                oneHot = np.zeros((len(testClasses), len(self.classes)))
                classesList = self.classes.tolist()
                for i in range(len(testClasses)):
                    oneHot[i][classesList.index(testClasses[i])] = 1
                self.backwardPass(oneHot.T)
            else:
                self.backwardPass(testClasses)

    def test(self, testData):
        #print("TEST DATA")
        #print(testData)
        self.network.setBatchSize(len(testData))
        classifications = ClassificationInfo.ClassificationInfo()
        testClasses = testData[self.classPlace].to_numpy()
        #print("TEST CLASSES")
        #print(testClasses)
        testData = testData.drop(columns=[self.classPlace])
        output = self.forwardPass(testData)
        if(self.classificationType == "classification"):
            output = output[0]
            for i in range(len(output)):
                #print("Predicted: " + str(output[i]) + " Actual: " + str(testClasses[i]))
                classifications.addTrueClass([testClasses[i], output[i]])
                classifications.addConfusion(self.classificationAccuracy(testClasses[i], output[i]))
                classifications.addLoss(self.losses)
        elif(self.classificationType == "regression"):
            #print("OUTPUT: ")
            #print(output)
            #print()
            for i in range(len(output[0])):
                #print("Predicted: " + str(output[0][i]) + " Actual: " + str(testClasses[i]))
                classifications.addTrueClass([testClasses[i], output[0][i]])
                classifications.addConfusion(self.regressionAccuracy(testClasses[i], output[0][i]))
                classifications.addLoss(self.losses)

        return classifications


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
        print("BACKWARD PASS...")
        #print("OUTPUT LAYER: ")
        currLayer = self.network.getOutputLayer()
        #print(currLayer.activations)
        #print("TEST CLASSES: ")
        #print(testClasses)
        #print("\nCALCULATE WEIGHT UPDATE FOR OUTPUT LAYER...")
        #calculate error (targets - predictions)
        error = testClasses - currLayer.activations
        errorAvg = np.mean(error, axis=0)
        #print("ERROR AVG")
        #print(errorAvg)
        self.losses.append(errorAvg)

        #print("MULTIPLY ERROR")
        #print(error)
        #print("BY PREV ACTIVATIONS")
        #print(currLayer.prev.activations.T)
        #print("PREVIOUS WEIGHTS:")
        #print(currLayer.prev.weights)

        # weight update for output layer
        outputWeightUpdate = self.learningRate * np.dot(error, currLayer.prev.activations.T) + self.momentum * currLayer.prev.prevUpdate

        #print("\nWEIGHT UPDATE:")
        #print(outputWeightUpdate)

        hiddenLayer = currLayer.getPrev()
        # if there are more than just the input and output layers...move through each layer and update weights
        while str(hiddenLayer.name) != str(self.network.getInputLayer().name):
            # apply hidden layer weight update rule
            #print("\nCALCULATE WEIGHT UPDATE  " + str(hiddenLayer.name) + " LAYER...")

            #print("PREVIOUS WEIGHTS: ")
            #print(hiddenLayer.prev.weights)

            #print("PROPAGATED ERROR...")
            #print("\nOUTPUTWEIGHTUPDATE")
            #print(outputWeightUpdate)
            #print("ERROR")
            #print(error)
            #print("HIDDEN LAYER ACTIVATIONS")
            #print(hiddenLayer.activations)

            #print("HIDDEN LAYER PREVIOUS ACTIVATIONS")
            #print(hiddenLayer.prev.activations)

            propagatedError = np.dot(hiddenLayer.weights.T, error) * hiddenLayer.activations * (
                    1 - hiddenLayer.activations)

            #print("PROPAGATED ERROR")
            #print(propagatedError)
            # calculate hidden layer weight update
            hiddenWeightUpdate = self.learningRate * np.dot(propagatedError, hiddenLayer.prev.activations.T) + self.momentum * hiddenLayer.prev.prevUpdate
            #print("\nWEIGHT UPDATE:")
            #print(hiddenWeightUpdate)

            # apply weight update
            hiddenLayer.prev.prevWeights = hiddenLayer.prev.weights
            hiddenLayer.prev.weights = hiddenLayer.prev.weights + hiddenWeightUpdate
            hiddenLayer.prev.prevUpdate = hiddenWeightUpdate
            #print("\nNEW WEIGHTS:")
            #print(hiddenLayer.prev.weights)

            # move to previous layer in network
            hiddenLayer = hiddenLayer.getPrev()

        # apply weight update to output layer weights
        #print("\nNEW WEIGHTS FOR OUTPUT:")
        currLayer.prev.prevWeights = currLayer.prev.weights
        currLayer.prev.weights = currLayer.prev.weights + outputWeightUpdate
        currLayer.prev.prevUpdate = outputWeightUpdate
        #print(currLayer.prev.weights)

    def run(self):
        count = 0
        classificationInfos = []
        for fold in self.folds:
            print("FOLD: ")
            print(fold)
            trainData = self.data.drop(fold.index)
            testData = fold
            self.train(trainData)
            classification = self.test(testData)
            classificationInfos.append(classification)

        return classificationInfos