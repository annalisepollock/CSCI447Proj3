import copy
import random

import sys
import numpy as np
import Network
import math

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
                 printSteps=False
                 ):
        # attributes used for convergence
        self.patience = 3
        self.windowSize = 3
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

        # initialize values for particle swarm optimization algorithm
        self.inertia = inertia
        self.cognitiveComponent = cognitiveComponent
        self.socialComponent = socialComponent

        # print for video
        self.printSteps = printSteps
    
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

        # find initial global best solution
        candidateFitnessValues = []
        bestCandidateIndex = 0
        for candidate in swarmPopulation:
            numBatches = 3
            # calculate fitness of offspring, determine if it is better than current candidate
            candidateFitness = self.helperCalculateFitness(candidate, numBatches, batches, batchIndex)
            candidateFitnessValues.append(candidateFitness)
        
        bestCandidateIndex = candidateFitnessValues.index(min(candidateFitnessValues))
        
        # get weights from global solution
        globalSolutionWeights = []
        if self.printSteps == True:
            print(f"Initial global best fitness: {min(candidateFitnessValues)}")
        for layer in population[bestCandidateIndex].getLayers():
            globalSolutionWeightLayer = layer.getWeights()
            globalSolutionWeights.append(globalSolutionWeightLayer)

        # while it hasn't converged
        while not self.checkConvergence():
            print("Not Converged")
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

                            # values for NaN/overflow prevention
                            previousWeight = swarmPopulation[particle][weights][weightRow][weight]
                            previousVelocity = swarmVelocities[particle][weights][weightRow][weight]

                            np.seterr(over='raise')
                            try:
                
                                # compute velocity of dimension of particle
                                swarmVelocities[particle][weights][weightRow][weight] = self.inertia*swarmVelocities[particle][weights][weightRow][weight] + self.cognitiveComponent*r1*(personalBest[particle][weights][weightRow][weight] - swarmVelocities[particle][weights][weightRow][weight]) + self.socialComponent*r2*(globalSolutionWeights[weights][weightRow][weight] - swarmVelocities[particle][weights][weightRow][weight])
                                # compute position of specific dimension of particle 
                                swarmPopulation[particle][weights][weightRow][weight] = swarmPopulation[particle][weights][weightRow][weight] + swarmVelocities[particle][weights][weightRow][weight]
                                if self.printSteps == True:
                                    print(f"Particle {particle}, Weight ({weights}, {weightRow}, {weight}):")
                                    print(f"  Velocity updated to: {swarmVelocities[particle][weights][weightRow][weight]}")
                                    print(f"  Position updated to: {swarmPopulation[particle][weights][weightRow][weight]}")
                            except FloatingPointError:
                                return population[bestCandidateIndex]
                            # roll back to previous values if new ones are NaN
                            if math.isnan(swarmVelocities[particle][weights][weightRow][weight]):
                                swarmVelocities[particle][weights][weightRow][weight] = previousVelocity
                            if math.isnan(swarmPopulation[particle][weights][weightRow][weight]):
                                swarmPopulation[particle][weights][weightRow][weight] = previousWeight
                    print("Postion Updated")
                    print("Velocities Updated")

                # update personalbest for this particle
                newNetwork = copy.deepcopy(population[particle])
                newLayers = newNetwork.getLayers()
                newWeights = []
                for weights in range(len(newLayers)):
                    newLayers[weights].setWeights(swarmPopulation[particle][weights])
                    newWeights.append(newLayers[weights].getWeights())
                # compare fitnesses of both old and new network
                newNetworkFitness = self.helperCalculateFitness(newWeights, numBatches, batches, batchIndex)
                oldNetworkFitness = self.helperCalculateFitness(swarmPopulation[particle], numBatches, batches, batchIndex)

                if oldNetworkFitness < newNetworkFitness:
                    population[particle] = newNetwork
                    swarmPopulation[particle] = newLayers
                    if self.printSteps == True:
                        print("Personal Best Updated")

                # update new global solution

                candidateFitnessValues = []
                bestCandidateIndex = 0
                for candidate in swarmPopulation:
                    numBatches = 3
                    # calculate fitness of each candidate, determine if it is better than current candidate
                    candidateFitness = self.helperCalculateFitness(candidate, numBatches, batches, batchIndex)
                    candidateFitnessValues.append(candidateFitness)

                bestCandidateIndex = candidateFitnessValues.index(min(candidateFitnessValues))
                if self.printSteps == True:
                    print("Global Best Fitness")
                    print(min(candidateFitnessValues))
                    print("Global Best Candidate")
                    print(bestCandidateIndex)
                    print(swarmPopulation[bestCandidateIndex])
                #print(bestCandidateIndex)
                #print(population[bestCandidateIndex].getLayers()[0].getWeights())
                
        # RETURN BEST INDIVIDUAL CANDIDATE SOLUTION
        candidateFitnessValues = []
        bestCandidateIndex = 0
        for candidate in swarmPopulation:
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
            candidateSolution = self.network.getWeights()
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
        self.network.setWeights(population[bestCandidateIndex])
        return self.network

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
        if not isinstance(self.network, Network.Network):
                raise TypeError("Network not of type Network")
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
            #print("\tloss for regression: " + str(loss))
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
        generations = 0
        while not self.checkConvergence():
            candidateFitnesses = []
            newPopulation = []
            totalLoss = 0
            #calculate fitness of each candidate solution
            #store in array with index to match candidate solutions
            for i in range(len(candidateSolutions)):
                sol = candidateSolutions[i]
                numBatches = 1
                candidateFitness = self.helperCalculateFitness(sol, numBatches, batches, batchIndex)
                batchIndex += numBatches
                candidateFitnesses.append(candidateFitness)
                totalLoss += candidateFitness

            avgLoss = totalLoss / len(candidateSolutions)
            self.losses.append(avgLoss)
            
            #create new population using selection, mutation, crossover
            while len(newPopulation) < len(candidateSolutions):
                #implement tounament selection - randomly select 10 % of the population and select the best candidate
                #print("\tSELECTING PARENTS")
                tournamentSize = int(0.1 * len(candidateSolutions))
                
                tournament = np.array(random.sample(candidateFitnesses, tournamentSize))
                parentIndice = candidateFitnesses.index(min(tournament))
                parent1 = candidateSolutions[parentIndice]

                if self.printSteps == True:
                    print("SELECTING TOURNAMENT HOLDS FITNESSES")
                    print(tournament)
                    print("SELECT LOWEST FITNESS VALUE")
                    print("Parent 1")
                    print(parent1)

                #repeat selection with new tournament
                tournament = np.array(random.sample(candidateFitnesses, tournamentSize))
                parentIndice = candidateFitnesses.index(min(tournament))
                parent2 = candidateSolutions[parentIndice]

                if self.printSteps == True:
                    print("SELECTING TOURNAMENT 2 HOLDS FITNESSES")
                    print(tournament)
                    print("SELECT LOWEST FITNESS VALUE")
                    print("Parent 2")
                    print(parent2)
                

                #implement crossover and mutation - randomly select a crossover point 
                #print("\tCROSSOVER AND MUTATION")
                if random.random() < self.geneticCrossoverRate:
                    child1, child2 = self.crossover(parent1, parent2)
                    #add children to new population
                    newPopulation.append(child1)
                    newPopulation.append(child2)
                else:
                    newPopulation.append(parent1)
                    newPopulation.append(parent2)
            
            candidateSolutions = newPopulation
            generations += 1
        
        # RETURN BEST INDIVIDUAL CANDIDATE SOLUTION
        candidateFitnessValues = []
        bestCandidateIndex = 0
        for candidate in candidateSolutions:
            numBatches = 3
            # calculate fitness of offspring, determine if it is better than current candidate
            candidateFitness = self.helperCalculateFitness(candidate, numBatches, batches, batchIndex)
            candidateFitnessValues.append(candidateFitness)

        bestCandidateIndex = candidateFitnessValues.index(min(candidateFitnessValues))
        self.network.setWeights(candidateSolutions[bestCandidateIndex])
        return self.network
                
    def crossover(self, parent1, parent2):
        #create networks to hold children
        child1 = []
        child2 = []

        parent1Network = copy.deepcopy(self.network)
        parent1Network.setWeights(parent1)
        parent2Network = copy.deepcopy(self.network)
        parent2Network.setWeights(parent2)
        if self.printSteps == True:
            print("IMPLEMENTING CROSSOVER")
            print()

        #loop through each layer in the parent networks
        for i in range(len(parent1Network.getLayers())):
            parent1Layer = parent1Network.getLayers()[i].getWeights()
            parent2Layer = parent2Network.getLayers()[i].getWeights()

            crossoverPoint = random.randint(0, parent1Layer.shape[1] -1)
            
            child1Layer = np.zeros(parent1Layer.shape)
            child2Layer = np.zeros(parent2Layer.shape)
            # Vectorized crossover
            child1Layer = np.where(np.arange(parent1Layer.shape[0])[:, None] <= crossoverPoint, parent1Layer, parent2Layer)
            child2Layer = np.where(np.arange(parent1Layer.shape[0])[:, None] <= crossoverPoint, parent2Layer, parent1Layer)
            if self.printSteps == True and i == 0:
                print("CROSSOVER POINT")
                print(crossoverPoint)
                print("PARENT 1 LAYER")
                print(parent1Layer)
                print("PARENT 2 LAYER")
                print(parent2Layer)
                print("CHILD 1 LAYER")
                print(child1Layer)
                print("CHILD 2 LAYER")
                print(child2Layer)
                print()
            # Vectorized mutation
            mutationMask = np.random.rand(*child1Layer.shape) < self.mutationRate
            mutationValues = np.random.uniform(-1, 1, child1Layer.shape)
            child1Layer += (mutationMask * mutationValues)
            child2Layer += (mutationMask * mutationValues)

            
            if self.printSteps == True and i == 0:
                print("MUTATION MASK")
                print(mutationMask)
                print("MUTATION VALUES")
                print(mutationValues)
                print("CHILD 1 LAYER AFTER MUTATION")
                print(child1Layer)
                print("CHILD 2 LAYER AFTER MUTATION")
                print(child2Layer)
                print()
            

            child1.append(child1Layer)
            child2.append(child2Layer)
                
        
        return child1, child2


