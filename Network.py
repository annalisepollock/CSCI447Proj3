from enum import Enum

class LayerName(Enum): 
    Input = 0
    Hidden = 1
    Output = 2

import Layer 

class Network:
    def __init__ (self, hiddenLayers, neuronsPerLayer, inputSize, outputSize, classes):
        self.layers = []
        #create input layer
        inputLayer = Layer.Layer(inputSize, neuronsPerLayer, LayerName.Input)
        self.layers.append(inputLayer)

        #create hidden layers
        for i in range(hiddenLayers):
            if(i == hiddenLayers - 1):
                hiddenLayer = Layer.Layer(neuronsPerLayer, outputSize, LayerName.Hidden)
            else:
                hiddenLayer = Layer.Layer(neuronsPerLayer, neuronsPerLayer, LayerName.Hidden)
            self.layers.append(hiddenLayer)
        
        #create output layer
        outputLayer = Layer.Layer(outputSize, 0, LayerName.Output, classes)
        self.layers.append(outputLayer)

        #connect layers
        for i in range(len(self.layers) - 1):
            self.layers[i].setNextLayer(self.layers[i + 1])
            self.layers[i + 1].setPreviousLayer(self.layers[i])
    
    def printNetwork(self): 
        for layer in self.layers:
            layer.printLayer()