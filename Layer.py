import numpy as np 
from Network import LayerName

class Layer:
    def __init__ (self, size, nextSize, name, batchSize, classes = [], classificationType=""):
        size = int(size)
        nextSize = int(nextSize)
        batchSize = int(batchSize)
        #INITIALIZE WEIGHTS BETWEEN SELF AND NEXT LAYER
        self.weights = np.random.uniform(-0.01, 0.01, (nextSize, size))
        self.prevWeights = np.zeros_like(self.weights) # to track convergence
        self.prevUpdate = np.zeros_like(self.weights)
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
        batchSize = int(batchSize)
        self.activations = np.empty((len(self.activations), batchSize))
        
    def getNext(self):
        return self.next
    def getPrev(self):
        return self.prev

    def setWeights(self, newWeights):
        self.weights = newWeights

    def getWeights(self):
        return self.weights
    
    def printLayer(self):
        print("Layer: " + str(self.name))
        print("Size: " + str(len(self.weights)))
        print("Weights: " + str(self.weights))
        print("Activations: " + str(self.activations))
        if(self.name == LayerName.Output):
            print("Classes: " + str(self.classes))
        print("\n")

    #APPLY SIGMOID ACTIVATION FUNCTION FOR HIDDEN LAYERS
    def sigmoid(self, values):
        return np.tanh(values)

    #APPLY SOFTMAX ACTIVATION FUNCTION FOR OUTPUT LAYER
    def softmaxActivation(self, values):
        max_value = np.max(values)
        exp_values = np.exp(values - max_value)
        sum_exp_values = np.sum(exp_values)

        return exp_values / sum_exp_values
    
    #takes a list of numpy arrays which are the updates 
    def forwardPass(self, nodeUpdates, printSteps=False):
        if(printSteps):
            print("Layer: " + str(self.name))
            print()
            print("Weights: ")
            print(self.weights)
            print()
            print("Node Updates: ")
            print(nodeUpdates)
            print()
        #RESET ACTIVATIONS
        for i in range(len(nodeUpdates)):
            self.activations[i] = nodeUpdates[i]

        if(self.name == LayerName.Output):
            newActivations = self.calculateOutput()
        #calculate values for the next layer
        elif(self.next.name == LayerName.Output):
            newActivations = np.array(np.dot(self.weights, self.activations))
        #If the next layer is a hidden layer, apply the sigmoid activation function
        else:
            newActivations = np.array(np.dot(self.weights, self.activations))
            newActivations = self.sigmoid(newActivations)

        if(printSteps):
            print("New Activations: ")
            print(newActivations)
            print()
        
        return newActivations
    
    def calculateOutput(self):
        if self.classificationType == "regression":
            return self.activations
        #for classification problems apply the softmax activation function
        elif self.classificationType == "classification":
            classifications = np.empty(len(self.activations[0]), dtype=object)
            for i in range(len(classifications)):
                self.activations[:, i] = self.softmaxActivation(self.activations[:, i])
                highestIndex = np.argmax(self.activations[:, i])
                classifications[i] = self.classes[highestIndex]
        #print(self.activations)
        #print()
        return classifications, self.activations
    