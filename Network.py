from enum import Enum
import numpy as np
import pandas as pd

class LayerName(Enum): 
    Input = 0
    Hidden = 1
    Output = 2

import Layer 

class Network:
    def __init__ (self, hiddenLayers, neuronsPerLayer, inputSize, outputSize, classificationType, batchSize, classes=[]):
        self.layers = []
        self.batchSize = int(batchSize)
        self.hiddenLayers = hiddenLayers
        self.neuronsPerLayer = neuronsPerLayer
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.classificationType = classificationType
        self.classes = classes

        #create input layer
        if  hiddenLayers == 0:
            self.inputLayer = Layer.Layer(inputSize, outputSize, LayerName.Input, batchSize, classes, classificationType)
            self.layers.append(self.inputLayer)
        else:
            self.inputLayer = Layer.Layer(inputSize, neuronsPerLayer, LayerName.Input, batchSize)
            self.layers.append(self.inputLayer)

            #create hidden layers
            for i in range(hiddenLayers):
                if i == hiddenLayers - 1:
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

    def reInitialize(self):
        self.__init__(self.hiddenLayers, self.neuronsPerLayer, self.inputSize, self.outputSize, self.classificationType, self.batchSize, self.classes)

    def setHiddenLayers(self, hiddenLayers):
        self.hiddenLayers = hiddenLayers
    
    def setBatchSize(self, batchSize):
        self.batchSize = int(batchSize)
        for layer in self.layers:
            layer.setBatchSize(batchSize)
    
    def getBatchSize(self):
        return self.batchSize

    def createBatches(self, data):
        batches = []
        #split data into an array of equal size batches
        for i in range(0, len(data), self.batchSize):
            if self.batchSize >= len(data[i:]):
                batches.append(data[i:])
            else:
                batches.append(data[i:i + self.batchSize])

        return batches

    def getLayers(self):
        # return array of layers in network
        return self.layers

    def getInputLayer(self):
        return self.inputLayer

    def getOutputLayer(self):
        return self.outputLayer

    def printNetwork(self):
        print("PRINTING NETWORK")
        print("Length layers: " + str(len(self.layers)))
        for layer in self.layers:
            layer.printLayer()
    
    def forwardPass(self, batch, printSteps=False):
        # split batch into features
        featureNames = batch.columns
        features = []
        #split data into arrays with values from each feature
        #this will be passed into input nodes 
        for feature in featureNames:
            features.append(batch[feature].values)
        if printSteps:
            print("SPLITTING BATCH INTO FEATURES")
            print("Features: ")
            print(features)
        features = np.array(features)
        for layer in self.layers:
            features = layer.forwardPass(features, printSteps)
        return features
    
    def getWeights (self):
        weights = []
        for layer in self.layers:
            weights.append(layer.getWeights())
        return weights
    
    #takes an matrix of weights for each layer
    def setWeights(self, weights):
        for i in range(len(self.layers)):
            self.layers[i].setWeights(weights[i])