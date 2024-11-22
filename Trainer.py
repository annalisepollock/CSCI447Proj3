import numpy as np
import Network

class Trainer:
    def __init__(self, algorithm, network, learningRate, momentum, batchSize ,classificationType, classPlace, trainData):
        # attributes used for convergence
        self.patience = 2
        self.windowSize = 1
        self.tolerance = 1e-1
        self.convergenceCount = 0
        # end attributes used for convergence

        # initialize values for genetic and differential evolution algorithms (values will be tuned)
        self.populationSize = 5
        self.scalingFactor = 1
        self.crossoverProbability = .8

        self.algorithm = algorithm
        self.network = network
        self.trainData = trainData
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.momentum = momentum
        self.classificationType = classificationType
        self.classPlace = classPlace
        self.classes = self.trainData[classPlace].unique()
        self.losses = []
    
    def train(self):
        if self.algorithm == "backpropagation":
            return self.backpropagation()
        elif self.algorithm == "swarmOptimization":
            return self.swarmOptimization()
        elif self.algorithm == "differentialEvolution":
            return self.differentialEvolution()
        elif self.algorithm == "geneticAlgorithm":
            return self.geneticAlgorithm()
        else:
            raise ValueError("Algorithm not recognized")
    
    def checkConvergence(self, printSteps = False):
        # customized hyperparameters for regression/classification
        if self.classificationType == 'regression':
            targetRange = self.trainData[self.classPlace].max() - self.trainData[self.classPlace].min()
            self.tolerance = max(self.tolerance * targetRange, 1e-5) # scale tolerance to range of target values
        else: # classification
            self.windowSize = 3

        if len(self.losses) < self.windowSize*2:
            if printSteps == True:
                print("NOT ENOUGH DATA TO CHECK CONVERGENCE")
            return False # not enough data to check convergence

        # Calculate moving averages for the last two windows
        recentAvg1 = np.mean(self.losses[-self.windowSize:])
        recentAvg2 = np.mean(self.losses[-2 * self.windowSize:-self.windowSize])

        # Check if the change in moving averages is below the tolerance
        if abs(recentAvg1 - recentAvg2) < self.tolerance:
            self.convergenceCount += 1
            # If this condition is met over 'patience' epochs, consider converged
            if self.convergenceCount >= self.patience:
                return True
        else:
            # Reset counter if loss change exceeds tolerance
            self.convergenceCount = 0

        return False

    def backpropagation (self, printSteps = False):
        if printSteps == True:
            print("PROPAGATING NETWORK")
        # reset losses for fold
        self.losses = []
        # if testing batch size will be different...
        if self.network.getBatchSize() != self.batchSize:
            self.network.setBatchSize(self.batchSize)
        # create batches with train data
        batches = self.network.createBatches(self.trainData)
        batchIndex = 0 
        if printSteps == True:
            print("BATCHES")
            print(batches)
            print("BATCHES LENGTH")
            print(len(batches))
            print("BATCH SIZE")
            print(self.batchSize)
            print("TRAINING DATA SIZE")
            print(self.trainData.shape)
            print()
        # train until convergence or the end of the batches
        while not self.checkConvergence(printSteps) and batchIndex != len(batches):
            if printSteps == True:
                print("HERE")
            batch = batches[batchIndex % len(batches)]
            if self.network.getBatchSize() != batch.shape[0]:
                self.network.setBatchSize(batch.shape[0])
            
            testClasses = batch[self.classPlace].to_numpy()
            testData = batch.drop(columns=[self.classPlace])
            # run forward pass to get guesses
            output = self.forwardPass(testData)
            if printSteps == True:
                print("OUTPUT: ")
                print(output)
                print()
            # one hot encode classification guesses
            # Run backward pass to update weights
            if(self.classificationType == "classification"):
                output = output[1]
                oneHot = np.zeros((len(testClasses), len(self.classes)))
                classesList = self.classes.tolist()
                for i in range(len(testClasses)):
                    oneHot[i][classesList.index(testClasses[i])] = 1
                self.backwardPass(oneHot.T, printSteps)
            else:
                if(printSteps == True):
                    print("Starting back pass")
                    print()
                self.backwardPass(testClasses, printSteps)
            batchIndex += 1
        return self.network
    
    def forwardPass(self, batch, printSteps = False):
        if printSteps == True:
            print("FORWARD PASS")
            print("BATCH")
            print(batch)
            print
        #run batch through network
        return self.network.forwardPass(batch, printSteps)
    
    def backwardPass(self, testClasses, printSteps=False):
        currLayer = self.network.getOutputLayer()

        # print for video
        if printSteps == True:
            print("BACKWARD PASS...")
            print("OUTPUT LAYER: ")
            print(currLayer.activations)
            print("TEST CLASSES: ")
            print(testClasses)
            print()
            print("\nCALCULATE WEIGHT UPDATE FOR OUTPUT LAYER...")

        # initialize error
        error = testClasses - currLayer.activations
        errorAvg = np.mean(error, axis=0)

        epsilon = 1e-10  # small value to avoid log(0)
        predictions = np.clip(currLayer.activations, epsilon, 1 - epsilon)  # clip values for numerical stability

        if self.classificationType == "classification":
            # Cross-Entropy Loss for Classification
            numSamples = testClasses.shape[1]
            loss = -(1 / numSamples) * (np.sum(np.log(predictions) * testClasses))
            if printSteps == True:
                print("ERROR AVG")
                print(errorAvg)
                print()
                print("LOSS FOR CLASSIFICATION: " + str(loss))
            
            self.losses.append(loss)
            if printSteps == True:
                print("APPENDING LOSS")
                print(self.losses)
        else:
            # Mean Squared Error for Regression
            if printSteps == True:
                print("ERROR AVG")
                print(errorAvg)
                print(testClasses.shape)
            print()
            print("target vals: ")
            print(testClasses)
            print("predictions: ")
            print(currLayer.activations[0])

            print("error for regression: " + str(error))
            loss = np.mean((predictions[0] - testClasses) ** 2)
            print("loss for regression: " + str(loss))
            if printSteps == True:
                print("LOSS FOR REGRESSION: " + str(loss))
            self.losses.append(loss)
            if printSteps == True:
                print("APPENDING LOSS")
                print(self.losses)

        # weight update for output layer
        outputWeightUpdate = self.learningRate * np.dot(error,
                                                        currLayer.prev.activations.T) + self.momentum * currLayer.prev.prevUpdate

        if printSteps == True:
            print("\nWEIGHT UPDATE:")
            print(outputWeightUpdate)

        hiddenLayer = currLayer.getPrev()

        # if there are more than just the input and output layers...move through each layer and update weights
        while str(hiddenLayer.name) != str(self.network.getInputLayer().name):
            # apply hidden layer weight update rule
            if printSteps == True:
                print("\nCALCULATE WEIGHT UPDATE  " + str(hiddenLayer.name) + " LAYER...")

                print("PREVIOUS WEIGHTS: ")
                print(hiddenLayer.prev.weights)

            propagatedError = np.dot(hiddenLayer.weights.T, error) * hiddenLayer.activations * (
                    1 - hiddenLayer.activations)
            error = propagatedError
            if printSteps == True:
                print("\nPROPAGATED ERROR:")
                print(propagatedError)
            # calculate hidden layer weight update
            hiddenWeightUpdate = self.learningRate * np.dot(propagatedError,
                                                            hiddenLayer.prev.activations.T) + self.momentum * hiddenLayer.prev.prevUpdate
            if printSteps == True:
                print("\nWEIGHT UPDATE:")
                print(hiddenWeightUpdate)

            # apply weight update
            hiddenLayer.prev.prevWeights = hiddenLayer.prev.weights
            hiddenLayer.prev.weights = hiddenLayer.prev.weights + hiddenWeightUpdate
            hiddenLayer.prev.prevUpdate = hiddenWeightUpdate
            if printSteps == True:
                print("\nNEW WEIGHTS:")
                print(hiddenLayer.prev.weights)
            # move to previous layer in network
            hiddenLayer = hiddenLayer.getPrev()

        # apply weight update to output layer weights
        currLayer.prev.prevWeights = currLayer.prev.weights
        currLayer.prev.weights = currLayer.prev.weights + outputWeightUpdate
        currLayer.prev.prevUpdate = outputWeightUpdate
        if printSteps == True:
            print("\nNEW WEIGHTS FOR OUTPUT:")
            print(currLayer.prev.weights)
    
    def swarmOptimization(self):
        pass

    def differentialEvolution(self):
        print("RUNNING DIFFERENTIAL EVOLUTION...")
        populations = [] # array of networks

        # randomly generate N populations
        for i in range(self.populationSize):
            candidateNetwork = self.network
            candidateNetwork.reInitialize()
            candidateSolution = candidateNetwork # network with new randomized weights
            candidateSolution.printNetwork()
            populations.append(candidateSolution)

        self.network = populations[0]
        return self.network
        # while not converged...
            # mutation
            # crossover
            # selection

    def geneticAlgorithm(self):
        pass