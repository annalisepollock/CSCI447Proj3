import Network

class Learner: 

    def __init__ (self, data, classificationType, classPlace):
        self.data = data
        self.classificationType = classificationType
        self.testingData = self.data.sample(frac=0.1)
        self.trainingData = self.data.drop(self.testingData.index)
        self.learningRate = 0.01
        self.momentum = 0.9
        self.hiddenLayers = 2
        self.neuronsPerLayer = 5
        self.batchSize = 4
        self.features = self.data.shape[1] - 1
        self.classes = self.data[classPlace].unique()
        self.network = Network.Network(self.hiddenLayers, self.neuronsPerLayer, self.features, len(self.classes), self.classes, self.classificationType, self.batchSize)
    
    def setNetwork(self, network): # remove later after testing
        self.network = network
    
    def setTestClass(self, testClass):
        self.testClass = testClass

    def tuneData(self):
        pass

    def crossValidate(self):
        pass    
    def train(self):
        pass    
    def test(self):
        pass
    def forwardPass(self, batch):
        pass

    def backwardPass(self):
        test = self.network
        print("BACKWARD PASS TESTING: ")
        test.printNetwork()

    def gradientDescent(self):
        pass
