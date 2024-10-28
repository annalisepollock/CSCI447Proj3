import Network

class Learner: 

    def _init_(self, data, classificationType, classPlace):
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
        self.network = Network.Network(self.hiddenLayers, self.neuronsPerLayer, self.features, len(self.classes), self.classes, self.classificationType)
    
    def tuneData():
        pass

    def crossValidate():
        pass    
    def train():
        pass    
    def test():
        pass
    def forwardPass(batch):
        pass

    def backwardPass():
        pass
    def gradientDescent():
        pass
