import Cleaner
import Network 
import Learner
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import warnings


def main():
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    cleaner = Cleaner.Cleaner()
    #IMPORT DATA SETS 
    print("BREAST CANCER")
    breastCancerData =  fetch_ucirepo(id=15)
    breastCancerDataFrame = pd.DataFrame(breastCancerData.data.original)
    breastCancerClean = cleaner.clean(breastCancerDataFrame, ['Sample_code_number'], 'Class')
    breastCancerTest = breastCancerClean.sample(frac=0.5)
    testLearner = Learner.Learner(breastCancerTest, "classification", "Class")
    classifications = testLearner.run()
    for classification in classifications:
        classification.printAccuracy()
        print()

    '''
    print("GLASS")
    glassData =  fetch_ucirepo(id=42)
    glassDataFrame = pd.DataFrame(glassData.data.original)
    glassClean = cleaner.clean(glassDataFrame, ['Id_number'], 'Type_of_glass')
    glassLearner = Learner.Learner(glassClean, "classification", 'Type_of_glass')
    hiddenLayers = 1
    neuronsPerLayer = glassClean.shape[1] - 7
    inputSize = glassClean.shape[1] - 1
    outputSize = glassClean['Type_of_glass'].nunique()
    classification = "classification"
    batchSize = 10
    classes = glassClean['Type_of_glass'].unique()
    classifications = glassLearner.run()
    classifications = testLearner.run()
    for classification in classifications:
        classification.printAccuracy()
        print()
    print()

    print("SOYBEAN")
    soybeanData =  fetch_ucirepo(id=91)
    soybeanDataFrame = pd.DataFrame(soybeanData.data.original)
    soybeanClean = cleaner.clean(soybeanDataFrame, [], 'class')
    soybeanLearner = Learner.Learner(soybeanClean, "classification", 'class')
    folds = soybeanLearner.getFolds()
    hiddenLayers = 1
    neuronsPerLayer = soybeanClean.shape[1] - 1
    inputSize = soybeanClean.shape[1] - 1
    outputSize = soybeanClean['class'].nunique()
    classification = "classification"
    batchSize = 10
    classes = soybeanClean['class'].unique()
    soybeanLeaner = Network.Network(hiddenLayers, neuronsPerLayer, inputSize, outputSize, classification, batchSize, classes)
    soybeanLearner.setNetwork(soybeanLeaner)
    soybeanLearner.run()
    print()
    
    
    print("ABALONE")
    abaloneData = fetch_ucirepo(id=1)
    abaloneDataFrame = pd.DataFrame(abaloneData.data.original)
    abaloneClean = cleaner.clean(abaloneDataFrame, [], 'Rings')
    abaloneLearner = Learner.Learner(abaloneClean, "regression", 'Rings')
    hiddenLayers = 1
    neuronsPerLayer = abaloneClean.shape[1] - 1
    inputSize = abaloneClean.shape[1] - 1
    outputSize = 1
    classification = "regression"
    batchSize = 10
    abaloneNetwork = Network.Network(hiddenLayers, neuronsPerLayer, inputSize, outputSize, classification, batchSize)
    abaloneLearner.setNetwork(abaloneNetwork)
    abaloneLearner.run()
    
    print("COMPUTER HARDWARE")
    computerHardwareData =  fetch_ucirepo(id=29)
    computerHardwareDataFrame = pd.DataFrame(computerHardwareData.data.original)
    computerClean = cleaner.clean(computerHardwareDataFrame, [], 'ERP')
    computerLearner = Learner.Learner(computerClean, "regression", 'ERP')
    hiddenLayers = 1
    neuronsPerLayer = computerClean.shape[1] - 1
    inputSize = computerClean.shape[1] - 1
    outputSize = 1
    classification = "regression"
    batchSize = 10
    computerLeaner = Network.Network(hiddenLayers, neuronsPerLayer, inputSize, outputSize, classification, batchSize)
    computerLearner.setNetwork(computerLeaner)
    computerLearner.run()
    print()

    print("FOREST FIRES")
    forestFiresData =  fetch_ucirepo(id=162)
    forestFiresDataFrame = pd.DataFrame(forestFiresData.data.original)
    forestClean = cleaner.clean(forestFiresDataFrame, [], 'area')
    forestLearner = Learner.Learner(forestClean, "regression", 'area')
    hiddenLayers = 1
    neuronsPerLayer = forestClean.shape[1] - 1
    inputSize = forestClean.shape[1] - 1
    outputSize = 1
    classification = "regression"
    batchSize = 10
    forestLeaner = Network.Network(hiddenLayers, neuronsPerLayer, inputSize, outputSize, classification, batchSize)
    forestLearner.setNetwork(forestLeaner)
    forestLearner.run()
    print()
    '''

main()