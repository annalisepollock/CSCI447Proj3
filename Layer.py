import numpy as np 
from Network import LayerName

class Layer:
    def __init__ (self, size, nextSize, name, batchSize, classes = [], classificationType=""):
        self.weights = np.random.uniform(-0.01, 0.01, (nextSize, size))
        self.activations = np.empty((size, batchSize))
        self.name = name
        self.classificationType = classificationType
        if(name == LayerName.Output):
            self.classes = classes

    def setNextLayer(self, nextLayer):
        self.next = nextLayer
    
    def setPreviousLayer(self, previousLayer):
        self.prev = previousLayer

    def setBatchSize(self, batchSize):
        self.activations = np.empty((len(self.activations), batchSize))
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

    def sigmoid(self, values):
        return np.tanh(values)

    def softmaxActivation(self, values):
        for i in range(len(values)):
            values[i] = np.exp(values[i]) / np.sum(np.exp(values))
        return values
    
    #takes a list of numpy arrays which are the updates 
    def forwardPass(self, nodeUpdates):
        #update activation values 
        print("Layer: " + str(self.name.name))
        print("Node Updates: ")
        print(nodeUpdates)
        print()
        for i in range(len(nodeUpdates)):
            self.activations[i] = nodeUpdates[i]


        if(self.name == LayerName.Output):
            newActivations = self.calculateOutput()
        #calculate values for the next layer
        elif(self.next.name == LayerName.Output):
            newActivations = np.array(np.dot(self.weights, self.activations))
            print("Next Level Activations: ")
            print(newActivations)
            print()
        else:
            newActivations = np.array(np.dot(self.weights, self.activations))
            print("Next Level Activations before Sigmoid: ")
            print(newActivations)
            print()
            newActivations = self.sigmoid(newActivations)
            print("Next Level Activations: ")
            print(newActivations)
            print()
        
        return newActivations
    
    def calculateOutput(self):
        if self.classificationType == "regression":
            return self.activations
        elif self.classificationType == "classification":
            classifications = np.empty(len(self.activations[0]), dtype=object)
            for i in range(len(classifications)):
                self.activations[:, i] = self.softmaxActivation(self.activations[:, i])
                highestIndex = np.argmax(self.activations[:, i])
                classifications[i] = self.classes[highestIndex]
        print(self.activations)
        print()
        return classifications, self.activations
    