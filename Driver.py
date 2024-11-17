import Cleaner
import Learner
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import warnings


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
    print("BREAST CANCER FOLD 0 HIDDEN LAYERS")
    for classification in breastCancerClassifications:
        classification.printAccuracy()
        print()
    breastCancerLearner.setHiddenLayers(1)
    breastCancerClassifications = breastCancerLearner.run()
    print("BREAST CANCER FOLD 1 HIDDEN LAYERS")
    for classification in breastCancerClassifications:
        classification.printAccuracy()
        print()
    
    breastCancerLearner.setHiddenLayers(2)
    breastCancerClassifications = breastCancerLearner.run()
    foldAccuracyTotal = 0
    print("BREAST CANCER FOLD 2 HIDDEN LAYERS")
    for classification in breastCancerClassifications:
        classification.printAccuracy()
        print()
        foldAccuracyTotal += (classification.TP + classification.TN)/(classification.TP + classification.TN + classification.FP + classification.FN)
    print("Average Accuracy: " + str(foldAccuracyTotal/10))
    print()
    
    
    print("GLASS")
    glassData = fetch_ucirepo(id=42)
    glassDataFrame = pd.DataFrame(glassData.data.original)
    glassClean = cleaner.clean(glassDataFrame, ['Id_number'], 'Type_of_glass')
    glassLearner = Learner.Learner(glassClean, "classification", 'Type_of_glass')
    glassClassifications = glassLearner.run()
    for classification in glassClassifications:
        classification.printAccuracy()
        print()
    glassLearner.setHiddenLayers(1)
    glassClassifications = glassLearner.run()
    for classification in glassClassifications:
        classification.printAccuracy()
        print()
    glassLearner.setHiddenLayers(2)
    glassClassifications = glassLearner.run()
    for classification in glassClassifications:
        classification.printAccuracy()
        print()
    
    print("SOYBEAN")
    soybeanData =  fetch_ucirepo(id=91)
    soybeanDataFrame = pd.DataFrame(soybeanData.data.original)
    soybeanClean = cleaner.clean(soybeanDataFrame, [], 'class')
    soybeanLearner = Learner.Learner(soybeanClean, "classification", 'class')
    soybeanClassifications = soybeanLearner.run()
    for classification in soybeanClassifications:
        classification.printAccuracy()
        print()
    soybeanLearner.setHiddenLayers(1)
    soybeanClassifications = soybeanLearner.run()
    for classification in soybeanClassifications:
        classification.printAccuracy()
        print()
    soybeanLearner.setHiddenLayers(2)
    soybeanClassifications = soybeanLearner.run()
    for classification in soybeanClassifications:
        classification.printAccuracy()
        print()

    
    soybeanLearner.setHiddenLayers(2)
    soybeanClassifications = soybeanLearner.run()
    for classification in soybeanClassifications:
        classification.printAccuracy()
        print()
    print()
    
    print("ABALONE")
    abaloneData = fetch_ucirepo(id=1)
    abaloneDataFrame = pd.DataFrame(abaloneData.data.original)
    abaloneClean = cleaner.clean(abaloneDataFrame, [], 'Rings')
    abaloneLearner = Learner.Learner(abaloneClean, "regression", 'Rings')
    abaloneClassifications = abaloneLearner.run()
    for classification in abaloneClassifications:
        classification.printAccuracy()
        print()
    abaloneLearner.setHiddenLayers(1)
    abaloneClassifications = abaloneLearner.run()
    for classification in abaloneClassifications:
        classification.printAccuracy()
        print()
    abaloneLearner.setHiddenLayers(2)
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
    computerClassifications = computerLearner.run()
    for classification in computerClassifications:
        classification.printAccuracy()
        print()
    computerLearner.setHiddenLayers(1)
    computerClassifications = computerLearner.run()
    for classification in computerClassifications:
        classification.printAccuracy()
        print()
    computerLearner.setHiddenLayers(2)
    computerClassifications = computerLearner.run()
    for classification in computerClassifications:
        classification.printAccuracy()
        print()
    
    '''
    print("FOREST FIRES")
    forestFiresData =  fetch_ucirepo(id=162)
    forestFiresDataFrame = pd.DataFrame(forestFiresData.data.original)
    forestClean = cleaner.clean(forestFiresDataFrame, [], 'area')
    forestLearner = Learner.Learner(forestClean, "regression", 'area')
    forestClassifications = forestLearner.run()
    for classification in forestClassifications:
        classification.printAccuracy()
        print()
    forestLearner.setHiddenLayers(1)
    forestClassifications = forestLearner.run()
    for classification in forestClassifications:
        classification.printAccuracy()
        print()
    forestLearner.setHiddenLayers(2)
    forestClassifications = forestLearner.run()
    for classification in forestClassifications:
        classification.printAccuracy()
        print()
    '''

    
    




main()