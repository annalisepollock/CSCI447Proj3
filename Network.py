from enum import Enum
import numpy as np

class LayerName(Enum): 
    Input = 0
    Hidden = 1
    Output = 2

import Layer 

class Network:
    def __init__ (self, hiddenLayers, neuronsPerLayer, inputSize, outputSize, classificationType, batchSize, classes=[]):
        self.layers = []
        self.batchSize = batchSize
        #create input layer
        self.inputLayer = Layer.Layer(inputSize, neuronsPerLayer, LayerName.Input, batchSize)
        self.layers.append(self.inputLayer)

        #create hidden layers
        for i in range(hiddenLayers):
            if(i == hiddenLayers - 1):
                hiddenLayer = Layer.Layer(neuronsPerLayer, outputSize, LayerName.Hidden, batchSize)
            else:
                hiddenLayer = Layer.Layer(neuronsPerLayer, neuronsPerLayer, LayerName.Hidden, batchSize)
            self.layers.append(hiddenLayer)
        
        #create output layer
        self.outputLayer = Layer.Layer(outputSize, 0, LayerName.Output, batchSize, classes, classificationType)
        self.layers.append(self.outputLayer)

        #connect layers
        for i in range(len(self.layers) - 1):
            self.layers[i].setNextLayer(self.layers[i + 1])
            self.layers[i + 1].setPreviousLayer(self.layers[i])

    def checkConvergence(self, tolerance=0):
        for layer in self.layers:
            difference = np.abs(layer.prevWeights - layer.weights) # neg/pos doesn't matter

            # tolerance default = 0 -> no difference permitted to be considered convergence
            # other options: 0.00005, 0.00003, 0.00008, 0.00001, etc.
            if np.any(difference <= tolerance): # if all activation values have changed within tolerance permitted...
                return True # converged
            else:
                return False # not converged

    def setHiddenLayers(self, hiddenLayers):
        self.hiddenLayers = hiddenLayers
    
    def setBatchSize(self, batchSize):
        self.batchSize = batchSize
        for layer in self.layers:
            layer.setBatchSize(batchSize)
    
    def getBatchSize(self):
        return self.batchSize

    def createBatches(self, data):
        batches = []
        for i in range(0, len(data), self.batchSize):
            batches.append(data[i:i + self.batchSize])
        print("Batches: ")
        print(batches) 
        return batches

    def getInputLayer(self):
        return self.inputLayer

    def getOutputLayer(self):
        return self.outputLayer

    def printNetwork(self):
        for layer in self.layers:
            layer.printLayer()
    
    def forwardPass(self, batch):
        # split batch into features
        featureNames = batch.columns
        features = []
        for feature in featureNames:
            features.append(batch[feature].values)
        features = np.array(features)
        print("Features: ")
        print(features)
        print()
        for layer in self.layers:
            features = layer.forwardPass(features)
        return features