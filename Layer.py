import numpy as np 
from Network import LayerName

class Layer:
    def __init__ (self, size, nextSize, name, classes = []):
        self.weights = np.random.uniform(-0.1, 0.1, (nextSize, size))
        self.activations = np.zeros(size)
        self.name = name
        if(name == LayerName.Output):
            self.classes = classes
    
    def setNextLayer(self, nextLayer):
        self.next = nextLayer
    
    def setPreviousLayer(self, previousLayer):
        self.prev = previousLayer

    def getNext(self):
        return self.next
    def getPrev(self):
        return self.prev 
    def getWeights(self):
        return self.weights
    
    def printLayer(self):
        print("Layer: " + str(self.name))
        print("Weights: " + str(self.weights))
        print("Activations: " + str(self.activations))
        if(self.name == LayerName.Output):
            print("Classes: " + str(self.classes))
        print("\n")