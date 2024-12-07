import copy
import random

import sys
import numpy as np
import Network

class Trainer:
    def __init__(self, 
                 algorithm, 
                 learner, 
                 network, 
                 learningRate, 
                 momentum, 
                 batchSize ,
                 classificationType, 
                 classPlace, 
                 trainData, 
                 population,
                 crossoverRate, 
                 mutationRate,
                 binomialCrossoverRate,
                 scalingFactor, 
                 inertia,
                 cognitiveComponent,
                 socialComponent,
                 ):
        # attributes used for convergence
        self.patience = 2
        self.windowSize = 1
        self.tolerance = 1e-1
        self.learner = learner
        self.convergenceCount = 0
        # end attributes used for convergence

        # initialize values for genetic and differential evolution algorithms (values will be tuned)
        self.populationSize = population
        self.scalingFactor = 10
        self.crossoverProbability = .8
        self.geneticCrossoverRate = crossoverRate
        self.mutationRate = mutationRate
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
            finishedNetwork =  self.backpropagation()
        elif self.algorithm == "swarmOptimization":
            finishedNetwork = self.swarmOptimization()
        elif self.algorithm == "differentialEvolution":
            finishedNetwork = self.differentialEvolution()
        elif self.algorithm == "geneticAlgorithm":
            finishedNetwork = self.geneticAlgorithm()
        else:
            raise ValueError("Algorithm not recognized")
        
        self.learner.setLosses(self.losses)
        return finishedNetwork
    
    def checkConvergence(self, printSteps = False):
        # customized hyperparameters for regression/classification
        if self.classificationType == 'regression':
            self.patience = 2
            targetRange = self.trainData[self.classPlace].max() - self.trainData[self.classPlace].min()
            self.tolerance = max(self.tolerance * targetRange, 1e-5) # scale tolerance to range of target values
        else: # classification
            self.windowSize = 1

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
        loss = self.helperCalculateLoss(currLayer.activations, testClasses, True)
        self.losses.append(loss)

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
        print("RUNNING PARTICLE SWARM OPTIMIZATION...")
        # networks that populations are derived from
        population = []
        # make population of particles, which contain 2d arrays of each layer of weights 
        swarmPopulation = []
        # particle velocities
        swarmVelocities = []
        # personal best weight for each individual weight
        personalBest = []
        # personal best set of weights for each particle (size of population, contains networks)
        personalBestForParticle = []
        # inertia, cognitive/social coefficients (will make hyperparameters later)
        w = .5
        c1 = 2
        c2 = 2
        # if testing batch size will be different...
        if self.network.getBatchSize() != self.batchSize:
            self.network.setBatchSize(self.batchSize)
        # create batches with train data
        batches = self.network.createBatches(self.trainData)
        batchIndex = 0

        # randomly generate N amount of particles
        for i in range(self.populationSize):
            particle = [] # array of weights from each layer of network
            # create a deep copy of the current network
            candidateSolution = copy.deepcopy(self.network)
            candidateSolution.reInitialize() # network with new initialized weights
            particleVelocities = []
            for layer in candidateSolution.getLayers():
                weightArray = layer.getWeights()
                particle.append(weightArray)
                # initialize velocities to 0
                particleVelocities.append(np.zeros(weightArray.shape))
            swarmPopulation.append(particle)
            swarmVelocities.append(particleVelocities)
            population.append(candidateSolution)
            personalBest.append(particle)
            personalBestForParticle.append(candidateSolution)

        # find initial global best solution
        candidateFitnessValues = []
        bestCandidateIndex = 0
        for candidate in population:
            numBatches = 3
            # calculate fitness of offspring, determine if it is better than current candidate
            candidateFitness = self.helperCalculateFitness(candidate, numBatches, batches, batchIndex)
            candidateFitnessValues.append(candidateFitness)
        
        bestCandidateIndex = candidateFitnessValues.index(min(candidateFitnessValues))
        
        # get weights from global solution
        globalSolutionWeights = []
        for layer in population[bestCandidateIndex].getLayers():
            globalSolutionWeightLayer = layer.getWeights()
            globalSolutionWeights.append(globalSolutionWeightLayer)

        print(swarmPopulation[0][0][0][0])
        print(personalBest[0][0][0][0])
        print(swarmVelocities[0][0][0][0])
        print(globalSolutionWeights[0][0][0])

        # while it hasn't converged
        while not self.checkConvergence():
            # for each particle
            for particle in range(len(swarmPopulation)):
                # for each set of weights in a layer
                for weights in range(len(swarmPopulation[particle])):
                    # for each row of weights in weight array
                    for weightRow in range(len(swarmPopulation[particle][weights])):
                        # for each single weight individually
                        for weight in range(len(swarmPopulation[particle][weights][weightRow])):
                            # compute each new velocity individually
                            # generate 2 random values for this function
                            r1 = random.random()
                            r2 = random.random()
                            swarmVelocities[particle][weights][weightRow][weight] = w*swarmVelocities[particle][weights][weightRow][weight] + c1*r1*(personalBest[particle][weights][weightRow][weight] - swarmVelocities[particle][weights][weightRow][weight]) + c2*r2*(globalSolutionWeights[weights][weightRow][weight] - swarmVelocities[particle][weights][weightRow][weight])
                            # compute position of specific dimension of particle 
                            swarmPopulation[particle][weights][weightRow][weight] = swarmPopulation[particle][weights][weightRow][weight] + swarmVelocities[particle][weights][weightRow][weight]
                # update personalbest for this particle
                newNetwork = copy.deepcopy(population[particle])
                newLayers = newNetwork.getLayers()
                for weights in range(len(newLayers)):
                    newLayers[weights].setWeights(swarmPopulation[particle][weights])
                # compare fitnesses of both old and new network
                newNetworkFitness = self.helperCalculateFitness(newNetwork, numBatches, batches, batchIndex)
                oldNetworkFitness = self.helperCalculateFitness(population[particle], numBatches, batches, batchIndex)

                if oldNetworkFitness < newNetworkFitness:
                    population[particle] = newNetwork

                # update new global solution

                candidateFitnessValues = []
                bestCandidateIndex = 0
                for candidate in population:
                    numBatches = 3
                    # calculate fitness of each candidate, determine if it is better than current candidate
                    candidateFitness = self.helperCalculateFitness(candidate, numBatches, batches, batchIndex)
                    candidateFitnessValues.append(candidateFitness)

                bestCandidateIndex = candidateFitnessValues.index(min(candidateFitnessValues))
                
        # RETURN BEST INDIVIDUAL CANDIDATE SOLUTION
        candidateFitnessValues = []
        bestCandidateIndex = 0
        for candidate in population:
            numBatches = 3
            # calculate fitness of each candidate, determine if it is better than current candidate
            candidateFitness = self.helperCalculateFitness(candidate, numBatches, batches, batchIndex)
            candidateFitnessValues.append(candidateFitness)

        bestCandidateIndex = candidateFitnessValues.index(min(candidateFitnessValues))
        return population[bestCandidateIndex]

    def differentialEvolution(self):
        print("RUNNING DIFFERENTIAL EVOLUTION...")

        # array of networks
        population = []
        # reset losses for fold
        self.losses = []
        # if testing batch size will be different...
        if self.network.getBatchSize() != self.batchSize:
            self.network.setBatchSize(self.batchSize)
        # create batches with train data
        batches = self.network.createBatches(self.trainData)
        batchIndex = 0

        # randomly generate N populations
        for i in range(self.populationSize):
            # create a deep copy of the current network (all new objects + references)
            candidateSolution = copy.deepcopy(self.network)
            candidateSolution.reInitialize() # network with new initialized weights
            candidateSolution.printNetwork()
            population.append(candidateSolution)

        # TO-DO: while not converged...
        h = 0
        while not self.checkConvergence():
            print("DIFF EVOLUTION ROUND " + str(h))
            # for each solution in the population...
            for i in range(len(population)):
                print("MUTATION ON " + str(i+1) + " CANDIDATE SOLUTION")
                sol = population[i]

                numBatches = 3
                candidateFitness = self.helperCalculateFitness(sol, numBatches, batches, batchIndex)
                batchIndex += numBatches

                # extract weights for each layer from the Layer objects
                solWeights = []

                for j in range(len(sol.getLayers())-1):
                    solutionLayer = sol.getLayers()[j].getWeights()
                    solWeights.append(solutionLayer)

                # MUTATION
                donor = []
                # randomly select three candidate solutions that are not the current solution
                x1, x2, x3 = random.sample([p for k, p in enumerate(population) if k != i], 3)

                # perform calculation for all sets of weights (exclude output layer because it will not hold weights)
                for j in range(len(sol.getLayers())-1):
                    layerDonor = x1.getLayers()[j].getWeights() + self.scalingFactor*(x2.getLayers()[j].getWeights() - x3.getLayers()[j].getWeights())
                    donor.append(layerDonor)

                # CROSSOVER
                offspring = []
                for candidateLayer, donorLayer in zip(solWeights, donor):
                    # Check that corresponding layers have matching shapes
                    assert candidateLayer.shape == donorLayer.shape, "Target and donor layers must have the same shape" # return error message if false

                    # Generate random mask and combine weights
                    binomialEval = np.random.rand(*candidateLayer.shape) <= self.crossoverProbability # is randomly generated value < alpha?
                    offspringLayer = np.where(binomialEval, candidateLayer, donorLayer)
                    offspring.append(offspringLayer)

                # create a network with the calculated offspring weights + biases
                offspringSolution = copy.deepcopy(self.network)
                offspringLayers = offspringSolution.getLayers()
                for i in range(len(offspringLayers)-1):
                    offspringLayers[i].setWeights(offspring[i])

                # calculate fitness of offspring, determine if it is better than current candidate
                offspringFitness = self.helperCalculateFitness(offspringSolution, numBatches, batches, batchIndex)
                batchIndex += numBatches

                # SELECTION
                if offspringFitness < candidateFitness:
                    population[i] = offspringSolution
                    self.losses.append(offspringFitness)
                else:
                    self.losses.append(candidateFitness)

                h += 1

        # RETURN BEST INDIVIDUAL CANDIDATE SOLUTION
        candidateFitnessValues = []
        bestCandidateIndex = 0
        for candidate in population:
            numBatches = 3
            # calculate fitness of offspring, determine if it is better than current candidate
            candidateFitness = self.helperCalculateFitness(candidate, numBatches, batches, batchIndex)
            candidateFitnessValues.append(candidateFitness)

        bestCandidateIndex = candidateFitnessValues.index(min(candidateFitnessValues))
        return population[bestCandidateIndex]

    def helperCalculateFitness(self, solution, numBatches, batches, batchIndex):
        # calculate loss numBatches times and return the average
        solLoss = []
        self.network.setWeights(solution)

        # calculate fitness of current candidate; pass a given # of batches in and calculate avg loss
        for k in range(numBatches):
            # get batch of data
            batch = batches[batchIndex % len(batches)]
            if self.network.getBatchSize() != batch.shape[0]:
                self.network.setBatchSize(batch.shape[0])

            batchClasses = batch[self.classPlace].to_numpy()
            batchData = batch.drop(columns=[self.classPlace])

            # test weights and biases on current
            predictions = self.network.forwardPass(batchData)
            targetValues = batchClasses

            # prepare output for loss calculation
            if (self.classificationType == "classification"):
                predictions = predictions[1]
                oneHot = np.zeros((len(batchClasses), len(self.classes)))
                classesList = self.classes.tolist()
                for i in range(len(batchClasses)):
                    oneHot[i][classesList.index(batchClasses[i])] = 1
                targetValues = oneHot.T

            outputLoss = self.helperCalculateLoss(predictions, targetValues, False)
            solLoss.append(outputLoss)
            batchIndex += 1

        return np.mean(solLoss)

    def helperCalculateLoss(self, output, targets, printSteps):
        loss = 0
        error = targets - output
        errorAvg = np.mean(error, axis=0)

        epsilon = 1e-10  # small value to avoid log(0)
        predictions = np.clip(output, epsilon, 1 - epsilon)  # clip values for numerical stability

        if self.classificationType == "classification":
            # Cross-Entropy Loss for Classification
            numSamples = targets.shape[1]
            loss = -(1 / numSamples) * (np.sum(np.log(predictions) * targets))
            if printSteps == True:
                print("ERROR AVG")
                print(errorAvg)
                print()
                print("LOSS FOR CLASSIFICATION: " + str(loss))

            if printSteps == True:
                print("APPENDING LOSS")
                print(self.losses)
        else:
            # Mean Squared Error for Regression
            if printSteps == True:
                print("ERROR AVG")
                print(errorAvg)
                print(targets.shape)
                print()
                print("target vals: ")
                print(targets)
                print("predictions: ")
                print(output[0])

                print("error for regression: " + str(error))
            loss = np.mean((predictions[0] - targets) ** 2)
            print("loss for regression: " + str(loss))
            if printSteps == True:
                print("LOSS FOR REGRESSION: " + str(loss))
            if printSteps == True:
                print("APPENDING LOSS")
                print(self.losses)

        return loss

    def geneticAlgorithm(self):
        print("RUNNING GENETIC ALGORITHM...")
        candidateSolutions = []
        self.losses = []

        if self.network.getBatchSize() != self.batchSize:
            self.network.setBatchSize(self.batchSize)
        # create batches with train data
        batches = self.network.createBatches(self.trainData)
        batchIndex = 0

        # randomly generate N populations
        for i in range(self.populationSize):
            # create a deep copy of the current network (all new objects + references)
            candidateSolution = self.network.getWeights()
            self.network.reInitialize() # network with new initialized weights
            candidateSolutions.append(candidateSolution)
        
        #Evaluate fitness of each candidate solution
        #To do: check convergence
        while not self.checkConvergence() and batchIndex < len(batches):
            candidateFitnesses = []
            newPopulation = []
            print("GENERATION " + str(i))
            print("Evaluating fitness of candidate solutions")
            for i in range(len(candidateSolutions)):
                sol = candidateSolutions[i]
                numBatches = 1
                candidateFitness = self.helperCalculateFitness(sol, numBatches, batches, batchIndex)
                batchIndex += numBatches
                candidateFitnesses.append(candidateFitness)
                self.losses.append(candidateFitness)
            
            #create new population using selection, mutation, crossover
            while len(newPopulation) < len(candidateSolutions):
                #implement tounament selection - randomly select 10 % of the population and select the best two candidates
                print("SELECTING PARENTS")
                tournamentSize = int(0.1 * len(candidateSolutions))

                tournament = np.array(random.sample(candidateFitnesses, tournamentSize))
                parentIndice = candidateFitnesses.index(min(tournament))
                parent1 = candidateSolutions[parentIndice]

                tournament = np.array(random.sample(candidateFitnesses, tournamentSize))
                parentIndice = candidateFitnesses.index(min(tournament))
                parent2 = candidateSolutions[parentIndice]
                

                #implement crossover and mutation - randomly select a crossover point 
                print("CROSSOVER AND MUTATION")
                if random.random() < self.geneticCrossoverRate:
                    child1, child2 = self.crossover(parent1, parent2)
                    #add children to new population
                    newPopulation.append(child1)
                    newPopulation.append(child2)
                else:
                    newPopulation.append(parent1)
                    newPopulation.append(parent2)
            
            candidateSolutions = newPopulation
        
        # RETURN BEST INDIVIDUAL CANDIDATE SOLUTION
        candidateFitnessValues = []
        bestCandidateIndex = 0
        for candidate in candidateSolutions:
            numBatches = 3
            # calculate fitness of offspring, determine if it is better than current candidate
            candidateFitness = self.helperCalculateFitness(candidate, numBatches, batches, batchIndex)
            candidateFitnessValues.append(candidateFitness)

        bestCandidateIndex = candidateFitnessValues.index(min(candidateFitnessValues))
        return candidateSolutions[bestCandidateIndex]
                
    def crossover(self, parent1, parent2):
        #create networks to hold children
        child1 = copy.deepcopy(self.network)
        child2 = copy.deepcopy(self.network)

        #loop through each layer in the parent networks
        for i in range(len(parent1.getLayers())):
            parent1Layer = parent1.getLayers()[i].getWeights()
            parent2Layer = parent2.getLayers()[i].getWeights()

            crossoverPoint = random.randint(0, parent1Layer.shape[1] -1)

            child1LayerWeights = np.zeros(parent1Layer.shape)
            child2LayerWeights = np.zeros(parent2Layer.shape)

            # Vectorized crossover
            child1LayerWeights = np.where(np.arange(parent1Layer.shape[0])[:, None] <= crossoverPoint, parent1Layer, parent2Layer)
            child2LayerWeights = np.where(np.arange(parent1Layer.shape[0])[:, None] <= crossoverPoint, parent2Layer, parent1Layer)

            # Vectorized mutation
            mutationMask = np.random.rand(*child1LayerWeights.shape) < self.mutationRate
            mutationValues = np.random.uniform(-1, 1, child1LayerWeights.shape)
            child1LayerWeights += mutationMask * mutationValues
            child2LayerWeights += mutationMask * mutationValues

            child1.getLayers()[i].setWeights(child1LayerWeights)
            child2.getLayers()[i].setWeights(child2LayerWeights)
                
        
        return child1, child2


