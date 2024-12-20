import Network
import math
import pandas as pd
import numpy as np
import Trainer
import ClassificationInfo
from ClassificationInfo import Accuracy

class Learner: 

    def __init__ (self, data, classificationType, classPlace, algorithm="differentialEvolution"):
        # convergence testing
        self.patience = 2
        self.windowSize = 1
        self.tolerance = 1e-1
        self.losses = []
        self.algorithm = algorithm
        self.convergenceCount = 0
        # end convergence testing
        self.testData = data.sample(frac=0.1)
        self.data = data.drop(self.testData.index)
        self.classPlace = classPlace
        self.classificationType = classificationType
        if( classificationType == "classification"):
            self.folds  = self.crossValidateClassification(self.data, classPlace)
        elif( classificationType == "regression"):
            self.folds = self.crossValidateRegression(self.data, classPlace)
            self.accuracyThreshold = 0.3 * data[classPlace].mean()
        else:
            raise ValueError("Invalid classification type")
        self.learningRate = -1 #backpropagation
        self.momentum = -1 #backpropagation
        self.crossoverRate = -1 #genetic algorithm
        self.mutationRate = -1 #genetic algorithm
        self.populationSize = 5 #GA DE PSO
        self.scalingFactor = -1 #DE
        self.binomialCrossoverProb = -1 #DE
        self.inertia = .7 #PSO
        self.cognitiveUpdateRate = 2 #PSO
        self.socialUpdateRate = 2 #PSO
        self.hiddenLayers = 1
        self.neuronsPerLayer = self.data.shape[0]
        self.batchSize = 10
        self.features = self.data.shape[1] - 1
        self.classes = self.data[classPlace].unique()
        self.network = self.tuneData()
        
    def setLosses(self, losses):
        self.losses = losses
    
    def setHiddenLayers(self, hiddenLayers):
        self.hiddenLayers = hiddenLayers
        self.resetNetwork()
        #self.network.printNetwork()
    def resetNetwork(self):
        if self.classificationType == "classification":
            outputSize = len(self.classes)
        else:
            outputSize = 1
        self.network = Network.Network(self.hiddenLayers, self.neuronsPerLayer, self.features, outputSize, self.classificationType, self.batchSize, self.classes)
    def getFolds(self):
        return self.folds
    
    def setTestClass(self, testClass):
        self.testClass = testClass
    
    def checkOscillation(self, losses, threshold=2):
        window_size = len(losses) // 2

        oscillations = 0
        #check for oscillations in a given window return false if over threshold 
        for i in range(1, window_size):
            if (losses[-i] > losses[-(i + 1)] and losses[-(i + 2)] < losses[-(i + 1)]) or \
           (losses[-i] < losses[-(i + 1)] and losses[-(i + 2)] > losses[-(i + 1)]):
                oscillations += 1
        return oscillations >= threshold
        
    def tuneData(self):
        if self.classificationType == "classification":
            outputSize = len(self.classes)
        else:
            outputSize = 1

        self.network = Network.Network(self.hiddenLayers, self.neuronsPerLayer, self.features, outputSize, self.classificationType, self.batchSize, self.classes)
        
        foldIndex = 0
        fold = self.folds[foldIndex % len(self.folds)]
        if self.algorithm == "backpropagation":
            self.tuneBackprop()
        
        else:
            if self.algorithm == "geneticAlgorithm":
                populationRange = np.linspace(50, 200, 6).astype(int)
                self.tunePopulationSize(populationRange)
                self.tuneGenetic()
            if self.algorithm == "differentialEvolution":
                #define population range
                populationRange = np.linspace(10, 20, 3).astype(int)
                self.tunePopulationSize(populationRange)
                self.tuneDifferentialEvolution()
            if self.algorithm == "swarmOptimization":
                #define population range
                self.tunePopulationSize([5, 10, 15, 20, 25])
                self.tuneParticleSwarm()
        
        
        nueronsPerLayer = self.data.shape[0]
        #set 5 possible nueron values with a max at the number of input values
        nueronValues = np.linspace(20, nueronsPerLayer, 5).astype(int)
        accuracy = 0
        bestNuerons = 0
        #test accuracy of each value
        for nuerons in nueronValues:
            self.neuronsPerLayer = nuerons
            self.network = Network.Network(self.hiddenLayers, self.neuronsPerLayer, self.features, outputSize, self.classificationType, self.batchSize, self.classes)
            nueronTrainer = Trainer.Trainer(self.algorithm, self, self.network, self.learningRate, self.momentum, self.batchSize,self.classificationType, self.classPlace, self.data.drop(fold.index), self.populationSize, self.crossoverRate, self.mutationRate, self.binomialCrossoverProb, self.scalingFactor, self.inertia, self.cognitiveUpdateRate, self.socialUpdateRate)
            self.network = nueronTrainer.train()
            output = self.test(self.testData)
            foldAccuracy = (output.getTP() + output.getTN()) / (output.getTP() + output.getTN() + output.getFP() + output.getFN())
            if(foldAccuracy > accuracy):
                accuracy = foldAccuracy
                bestNuerons = nuerons
            foldIndex += 1
        self.neuronsPerLayer = bestNuerons
        if self.neuronsPerLayer == 0:
            self.neuronsPerLayer = self.data.shape[0] - 10
        
        #tune batch size
        #Ensure dataset not super small
        trainSize = self.data.drop(self.folds[foldIndex % len(self.folds)].index).shape[0]
        batchSizes = [int(trainSize * 0.05), int(trainSize * 0.8), int(trainSize * 0.12), int(trainSize * 0.17), int(trainSize * 0.2), int(trainSize * 0.25)]
        accuracy = 0
        bestBatchSize = 0
        for batchSize in batchSizes:
            self.batchSize = batchSize
            self.network = Network.Network(self.hiddenLayers, self.neuronsPerLayer, self.features, outputSize, self.classificationType, self.batchSize, self.classes)
            batchTrainer = Trainer.Trainer(self.algorithm, self, self.network, self.learningRate, self.momentum, self.batchSize,self.classificationType, self.classPlace, self.data.drop(fold.index), self.populationSize, self.crossoverRate, self.mutationRate, self.binomialCrossoverProb, self.scalingFactor, self.inertia, self.cognitiveUpdateRate, self.socialUpdateRate)
            self.network = batchTrainer.train()
            output = self.test(self.testData)
            foldAccuracy = (output.getTP() + output.getTN()) / (output.getTP() + output.getTN() + output.getFP() + output.getFN())
            if(foldAccuracy > accuracy):
                accuracy = foldAccuracy
                bestBatchSize = batchSize
            foldIndex += 1
        self.batchSize = bestBatchSize
        if self.batchSize == 0:
            self.batchSize = int(self.data.shape[0] / 10)
        
        return Network.Network(0, self.neuronsPerLayer, self.features, outputSize, self.classificationType, self.batchSize, self.classes)
    
    def tunePopulationSize(self, populationRange):
        foldIndex = 0
        #TUNE POPULATION SIZE
        #set 5 possible nueron values with a max at the number of input values
        accuracy = 0
        self.mutation = 0.1
        self.crossoverRate = 0.9
        bestPopulation = 15
        for population in populationRange:
            fold = self.folds[foldIndex % len(self.folds)]
            self.populationSize = population
            populationTrainer = Trainer.Trainer(self.algorithm, self, self.network, self.learningRate, self.momentum, self.batchSize,self.classificationType, self.classPlace, self.data.drop(fold.index), self.populationSize, self.crossoverRate, self.mutationRate, self.binomialCrossoverProb, self.scalingFactor, self.inertia, self.cognitiveUpdateRate, self.socialUpdateRate)
            self.network = populationTrainer.train()
            output = self.test(self.testData)
            foldAccuracy = (output.getTP() + output.getTN()) / (output.getTP() + output.getTN() + output.getFP() + output.getFN())
            if(foldAccuracy > accuracy):
                accuracy = foldAccuracy
                bestPopulation = population
            foldIndex += 1
        self.populationSize = bestPopulation
        
    def tuneBackprop(self):
        self.momentum = 0.9
        self.learningRate = 0.1
        foldIndex = 0
        momentumSet = False
        learningRateSet = False
        while not momentumSet:
            print("training momentum", str(self.momentum))
            fold = self.folds[foldIndex % len(self.folds)]
            if self.momentum <= 0.5:
                momentumSet = True
                self.momentum = 0.5
                break
            momentumTrainer = Trainer.Trainer(self.algorithm, self, self.network, self.learningRate, self.momentum, self.batchSize,self.classificationType, self.classPlace, self.data.drop(fold.index), self.populationSize, self.crossoverRate, self.mutationRate, self.binomialCrossoverProb, self.scalingFactor, self.inertia, self.cognitiveUpdateRate, self.socialUpdateRate)
            self.network = momentumTrainer.train()
            print("CURRENT LOSSES: ")
            print(self.losses)
            print("OSCILLATION:", self.checkOscillation(self.losses))
            if self.checkOscillation(self.losses):
                self.momentum -= 0.1
            else:
                momentumSet = True
            foldIndex += 1
        #TUNING LEARNING RATE
        #decrease as loss is oscillating
        while not learningRateSet:
            if self.learningRate == 0.0001:
                learningRateSet = True
                break
            learningRateTrainer = Trainer.Trainer(self.algorithm, self, self.network, self.learningRate, self.momentum, self.batchSize,self.classificationType, self.classPlace, self.data.drop(fold.index), self.populationSize, self.crossoverRate, self.mutationRate, self.binomialCrossoverProb, self.scalingFactor, self.inertia, self.cognitiveUpdateRate, self.socialUpdateRate)
            self.network = learningRateTrainer.train()
            if self.checkOscillation(self.losses):
                self.learningRate = self.learningRate / 10
            else:
                learningRateSet = True
            foldIndex += 1
    
    def tuneGenetic(self):
        #TUNE CROSSOVER RATE
        #set 5 possible nueron values with a max at the number of input values
        foldIndex = 0
        crossoverValues = [0.6, 0.7, 0.8, 0.9, 0.99]
        accuracy = 0
        self.mutationRate = 0.1
        #test accuracy of each value
        for crossover in crossoverValues:
            fold = self.folds[foldIndex % len(self.folds)]
            self.crossoverRate = crossover
            crossoverTrainer = Trainer.Trainer(self.algorithm, self, self.network, self.learningRate, self.momentum, self.batchSize,self.classificationType, self.classPlace, self.data.drop(fold.index), self.populationSize, self.crossoverRate, self.mutationRate, self.binomialCrossoverProb, self.scalingFactor, self.inertia, self.cognitiveUpdateRate, self.socialUpdateRate)
            self.network = crossoverTrainer.train()
            output = self.test(self.testData)
            foldAccuracy = (output.getTP() + output.getTN()) / (output.getTP() + output.getTN() + output.getFP() + output.getFN())
            if(foldAccuracy > accuracy):
                accuracy = foldAccuracy
                bestCrossover = crossover
            foldIndex += 1
        self.crossoverRate = bestCrossover
        #TUNE MUTATION RATE
        #set 5 possible nueron values with a max at the number of input values
        mutationValues = [0.01, 0.1, 0.2, 0.25, 0.3]
        accuracy = 0
        for mutation in mutationValues:
            fold = self.folds[foldIndex % len(self.folds)]
            self.mutationRate = mutation
            mutationTrainer = Trainer.Trainer(self.algorithm, self, self.network, self.learningRate, self.momentum, self.batchSize,self.classificationType, self.classPlace, self.data.drop(fold.index), self.populationSize, self.crossoverRate, self.mutationRate, self.binomialCrossoverProb, self.scalingFactor, self.inertia, self.cognitiveUpdateRate, self.socialUpdateRate)
            self.network = mutationTrainer.train()
            output = self.test(self.testData)
            foldAccuracy = (output.getTP() + output.getTN()) / (output.getTP() + output.getTN() + output.getFP() + output.getFN())
            if(foldAccuracy > accuracy):
                accuracy = foldAccuracy
                bestMutation = mutation
            foldIndex += 1
        self.mutationRate = bestMutation

    def tuneDifferentialEvolution(self):
        print("TUNE DIFFERENTIAL EVOLUTION")
        foldIndex = 0
        accuracy = 0

        # TUNE SCALING FACTOR
        bestScalingFactor = 0
        scalingFactorValues = np.linspace(0, 2, 5)
        # test accuracy of each value
        for val in scalingFactorValues:
            fold = self.folds[foldIndex % len(self.folds)]
            self.scalingFactor = val
            scalingFactorTrainer = Trainer.Trainer(self.algorithm, self, self.network, self.learningRate, self.momentum,
                                                   self.batchSize, self.classificationType, self.classPlace,
                                                   self.data.drop(fold.index), self.populationSize, self.crossoverRate,
                                                   self.mutationRate, self.binomialCrossoverProb, self.scalingFactor,
                                                   self.inertia, self.cognitiveUpdateRate, self.socialUpdateRate)
            self.network = scalingFactorTrainer.train()
            output = self.test(self.testData)
            foldAccuracy = (output.getTP() + output.getTN()) / (
                        output.getTP() + output.getTN() + output.getFP() + output.getFN())
            if (foldAccuracy > accuracy):
                accuracy = foldAccuracy
                bestScalingFactor = val
            foldIndex += 1
        self.scalingFactor = bestScalingFactor
        # TUNE BINOMIAL CROSSOVER PROBABILITY
        accuracy = 0  # reset accuracy measure
        bestProbability = 0
        probabilityValues = np.linspace(0, 1, 6, endpoint=False)
        # test accuracy of each value
        for val in probabilityValues:
            fold = self.folds[foldIndex % len(self.folds)]
            self.binomialCrossoverProb = val
            scalingFactorTrainer = Trainer.Trainer(self.algorithm, self, self.network, self.learningRate, self.momentum,
                                                   self.batchSize, self.classificationType, self.classPlace,
                                                   self.data.drop(fold.index), self.populationSize, self.crossoverRate,
                                                   self.mutationRate, self.binomialCrossoverProb, self.scalingFactor,
                                                   self.inertia, self.cognitiveUpdateRate, self.socialUpdateRate)
            self.network = scalingFactorTrainer.train()
            output = self.test(self.testData)
            foldAccuracy = (output.getTP() + output.getTN()) / (
                        output.getTP() + output.getTN() + output.getFP() + output.getFN())
            if (foldAccuracy > accuracy):
                accuracy = foldAccuracy
                bestProbability = val
            foldIndex += 1
        self.binomialCrossoverProb = bestProbability
        print("AFTER TUNING DIFFERENTIAL EVOLUTION, SCALING FACTOR: " + str(self.scalingFactor) + ", CROSSOVER PROBABILITY: " + str(
            self.binomialCrossoverProb))

    def tuneParticleSwarm(self):
        #TUNE INERTIA
        #TUNE COGNITIVE UPDATE RATE
        #TUNE SOCIAL UPDATE RATE
        pass

    def getNetwork(self):
        return self.network

    def crossValidateClassification(self, cleanDataset, classColumn ,printSteps = False):
         # 10-fold cross validation with stratification of classes
        if printSteps == True:
            print("Running cross calidation with stratification...")
        dataChunks = [None] * 10
        classes = np.unique(cleanDataset[classColumn])
        dataByClass = dict()

        for uniqueVal in classes:
        # Subset data based on unique class values
            classSubset = cleanDataset[cleanDataset[classColumn] == uniqueVal]
            dataByClass[uniqueVal] = classSubset

            numRows = math.ceil(classSubset.shape[0] / 10) # of class instances per fold

            for i in range(9):
                if(classSubset.shape[0] < numRows):
                    classChunk = classSubset
                else:
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

        return dataChunks

    def crossValidateRegression(self, data, targetColumn, printSteps = False):
        if printSteps == True:
            print("Running cross validation with stratification...")
        dataChunks = [None] * 10
        # Split data into 10 bins based on target column
        binLabels, binEdges = pd.qcut(data[targetColumn], q=10, retbins=True, labels=False, duplicates='drop')
        uniqueBins = np.unique(binLabels)
        #Pull even amounts of data from each bin for each fold
        for binLabel in uniqueBins:
            binSubset = data[binLabels == binLabel]
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
    
    def test(self, testData):
        #update batch size
        self.network.setBatchSize(len(testData))
        classifications = ClassificationInfo.ClassificationInfo()
        testClasses = testData[self.classPlace].to_numpy()
        testData = testData.drop(columns=[self.classPlace])
        #run forward pass on test data to get guesses
        output = self.forwardPass(testData)
        #append classification info
        if(self.classificationType == "classification"):
            output = output[0]
            for i in range(len(output)):
                classifications.addTrueClass([testClasses[i], output[i]])
                classifications.addConfusion(self.classificationAccuracy(testClasses[i], output[i]))
        elif(self.classificationType == "regression"):
            for i in range(len(output[0])):
                classifications.addTrueClass([testClasses[i], output[0][i]])
                classifications.addConfusion(self.regressionAccuracy(testClasses[i], output[0][i]))
        
        classifications.setLoss(self.losses)
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
    
    #define true positives and negatives for regression
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

    def forwardPass(self, batch, printSteps = False):
        if printSteps == True:
            print("FORWARD PASS")
            print("BATCH")
            print(batch)
            print
        #run batch through network
        return self.network.forwardPass(batch, printSteps)

    def run(self, printSteps = False):
        #run each fold and return classification info
        classificationInfos = []
        foldCount = 0
        for fold in self.folds:
            self.resetNetwork()
            trainData = self.data.drop(fold.index)
            testData = fold
            trainer = Trainer.Trainer(self.algorithm, self, self.network, self.learningRate, self.momentum, self.batchSize,self.classificationType, self.classPlace, trainData, self.populationSize, self.crossoverRate, self.mutationRate, self.binomialCrossoverProb, self.scalingFactor, self.inertia, self.cognitiveUpdateRate, self.socialUpdateRate, (printSteps and foldCount == 0))
            self.network = trainer.train()
            classification = self.test(testData)
            classificationInfos.append(classification)
            if printSteps == True and foldCount == 0:
                classification.printAccuracy()
            foldCount+=1

            

        return classificationInfos