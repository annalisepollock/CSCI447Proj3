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
        self.loss = []

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

    def printAccuracy(self):
        print("True Positives: " + str(self.TP))
        print("True Negatives: " + str(self.TN))
        print("False Positives: " + str(self.FP))
        print("False Negatives: " + str(self.FN))
        print("Loss: " + str(self.loss))

    def addLoss(self, loss):
        self.loss = loss
    def getLoss(self):
        return self.loss
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
