import AlgorithmAccuracy
import Cleaner
import Network
import Learner
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import warnings
import ClassificationInfo
import matplotlib.pyplot as plt


def main():
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    
    cleaner = Cleaner.Cleaner()
    '''
    #IMPORT DATA SETS 
    print("BREAST CANCER")
    breastCancerData = fetch_ucirepo(id=15)
    breastCancerDataFrame = pd.DataFrame(breastCancerData.data.original)
    breastCancerClean = cleaner.clean(breastCancerDataFrame, ['Sample_code_number'], 'Class')
    breastCancerTest = breastCancerClean.sample(frac=0.5)
    breastCancerLearner = Learner.Learner(breastCancerTest, "classification", "Class")

    breastCancerClassifications = breastCancerLearner.run()
    count = 0
    print("BREAST CANCER FOLD 0 HIDDEN LAYERS")
    for classification in breastCancerClassifications:
        if count == 0:
            classification.printAccuracy()
            count += 1
            print()
    breastCancerLearner.setHiddenLayers(1)
    breastCancerClassifications = breastCancerLearner.run()
    count = 0
    print("BREAST CANCER FOLD 1 HIDDEN LAYERS")
    for classification in breastCancerClassifications:
        if count == 0:
            classification.printAccuracy()
            count += 1
            print()
    
    breastCancerLearner.setHiddenLayers(2)
    breastCancerClassifications = breastCancerLearner.run(True)
    count = 0
    foldAccuracyTotal = 0
    print("BREAST CANCER FOLD 2 HIDDEN LAYERS")
    for classification in breastCancerClassifications:
        if count == 0:
            classification.printAccuracy()
            count += 1
            print()
        foldAccuracyTotal += (classification.TP + classification.TN)/(classification.TP + classification.TN + classification.FP + classification.FN)
    print("Average Accuracy: " + str(foldAccuracyTotal/10))
    print()
    
    print("GLASS")
    glassData = fetch_ucirepo(id=42)
    glassDataFrame = pd.DataFrame(glassData.data.original)
    glassClean = cleaner.clean(glassDataFrame, ['Id_number'], 'Type_of_glass')
    glassLearner = Learner.Learner(glassClean, "classification", 'Type_of_glass')
    classifications = glassLearner.run()
    for classification in classifications:
        classification.printAccuracy()
        print()


    print("SOYBEAN")
    soybeanData =  fetch_ucirepo(id=91)
    soybeanDataFrame = pd.DataFrame(soybeanData.data.original)
    soybeanClean = cleaner.clean(soybeanDataFrame, [], 'class')
    soybeanLearner = Learner.Learner(soybeanClean, "classification", 'class')
    soybeanLearner.setHiddenLayers(2)

    
    soybeanLearner.setHiddenLayers(2)
    soybeanClassifications = soybeanLearner.run()
    for classification in soybeanClassifications:
        classification.printAccuracy()
        print()
    print()
    
    '''
    print("ABALONE")
    abaloneData = fetch_ucirepo(id=1)
    abaloneDataFrame = pd.DataFrame(abaloneData.data.original)
    abaloneClean = cleaner.clean(abaloneDataFrame, [], 'Rings')
    abaloneLearner = Learner.Learner(abaloneClean, "regression", 'Rings')
    abaloneClassifications = abaloneLearner.run()
    for classification in abaloneClassifications:
        classification.printAccuracy()
        print()
    ''' 
    

    print("COMPUTER HARDWARE")
    computerHardwareData =  fetch_ucirepo(id=29)
    computerHardwareDataFrame = pd.DataFrame(computerHardwareData.data.original)
    computerClean = cleaner.clean(computerHardwareDataFrame, [], 'ERP')
    computerLearner = Learner.Learner(computerClean, "regression", 'ERP')
    computerLearner.setHiddenLayers(2)
    

    print("FOREST FIRES")
    forestFiresData =  fetch_ucirepo(id=162)
    forestFiresDataFrame = pd.DataFrame(forestFiresData.data.original)
    forestClean = cleaner.clean(forestFiresDataFrame, [], 'area')
    forestLearner = Learner.Learner(forestClean, "regression", 'area')


    classifications = forestLearner.run()
    count = 0
    print("FOREST FIRES FOLD 0 HIDDEN LAYERS")
    for classification in classifications:
        if count == 0:
            classification.printAccuracy()
            count += 1
            print()
    forestLearner.setHiddenLayers(1)
    classifications = forestLearner.run()
    count = 0
    print("FOREST FIRES FOLD 1 HIDDEN LAYERS")
    for classification in classifications:
        if count == 0:
            classification.printAccuracy()
            count += 1
            print()
    
    forestLearner.setHiddenLayers(2)
    classifications = forestLearner.run(True)
    count = 0
    foldAccuracyTotal = 0
    print("FOREST FIRES FOLD 2 HIDDEN LAYERS")
    for classification in classifications:
        if count == 0:
            classification.printAccuracy()
            count += 1
            print()
        foldAccuracyTotal += (classification.TP + classification.TN)/(classification.TP + classification.TN + classification.FP + classification.FN)
    print("Average Accuracy: " + str(foldAccuracyTotal/10))
    '''
    
    




main()