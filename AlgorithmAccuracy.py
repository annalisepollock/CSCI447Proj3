import ClassificationInfo

class AlgorithmAccuracy:
    def __init__(self, stats, numFeatures, name):
        if not isinstance(stats, ClassificationInfo.ClassificationInfo):
            raise TypeError('stats must be an instance of ClassificationInfo')
        self.stats = stats
        #calculate precision, recall, f1
        if(self.stats.TP + self.stats.FP) == 0:
            self.precision = 0
        else:
            self.precision = (self.stats.TP)/(self.stats.TP + self.stats.FP)
        if(self.stats.TP + self.stats.FN) == 0:
            self.recall = 0
        else:
            self.recall = (self.stats.TP)/(self.stats.TP + self.stats.FN)
        
        self.f1 = 0

        if (self.precision + self.recall) != 0:
            self.f1 = 2 * ((self.precision * self.recall)/(self.precision + self.recall))

        self.totalAccuracy = self.calculateTotalAccuracy()
        self.numFeatures = numFeatures
        self.name = name
    def confusionMatrix(self):
        return {"TP": self.stats.TP, "FP": self.stats.FP, "FN": self.stats.FN, "TN": self.stats.TN}
    
    #calculate loss by summing the number of correct classifications and dividing by the total number of classifications
    def calculateTotalAccuracy(self):
        return (self.stats.TP + self.stats.TN)/(self.stats.TP + self.stats.TN + self.stats.FP + self.stats.FN)
    
    def getF1(self):
        return self.f1
    
    def getTotalAccuracy(self):
        return self.totalAccuracy
    
    def getNumFeatures(self):
        return self.numFeatures

    def getName(self):
        return self.name
    
    def print(self):
        print("DataSet: " + self.name)
        print("F1: " + str(self.f1))
        print("Total Accuracy: " + str(self.totalAccuracy))
        print("Confusion Matrix: " + str(self.confusionMatrix()))
        print("Number of Features: " + str(self.numFeatures))