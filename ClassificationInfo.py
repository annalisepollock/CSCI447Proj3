from enum import Enum
class Accuracy(Enum):
    TP = 1
    TN = 2
    FP = 3
    FN = 4

class ClassificationInfo:
    def __init__(self):
        self.trueClasses = [] # [[trueClass, AssignedClass], [trueClass, AssignedClass], ...]
        self.FP = 0
        self.FN = 0
        self.TP = 0
        self.TN = 0

    def addTrueClass(self, trueClass):
        self.trueClasses.append(trueClass)
    
    #increment confusion as classifications are made
    def addConfusion(self, Accuracy):
        if Accuracy == Accuracy.TP:
            self.TP += 1
        elif Accuracy == Accuracy.TN:
            self.TN += 1
        elif Accuracy == Accuracy.FP:
            self.FP += 1
        elif Accuracy == Accuracy.FN:
            self.FN += 1
    
    def getFP(self):
        return self.FP
    def getFN(self):
        return self.FN
    def getTP(self):
        return self.TP
    def getTN(self):
        return self.TN
    def getTrueClasses(self): 
        return self.trueClasses
