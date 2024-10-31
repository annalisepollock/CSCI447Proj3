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

    print("GLASS")
    glassData =  fetch_ucirepo(id=42)
    glassDataFrame = pd.DataFrame(glassData.data.original)
    glassClean = cleaner.clean(glassDataFrame, ['Id_number'], 'Type_of_glass')

    print("SOYBEAN")
    soybeanData =  fetch_ucirepo(id=91)
    soybeanDataFrame = pd.DataFrame(soybeanData.data.original)
    soybeanClean = cleaner.clean(soybeanDataFrame, [], 'class')

    print("ABALONE")
    abaloneData = fetch_ucirepo(id=1)
    abaloneDataFrame = pd.DataFrame(abaloneData.data.original)
    abaloneClean = cleaner.clean(abaloneDataFrame, [], 'Rings')
    

    print("COMPUTER HARDWARE")
    computerHardwareData =  fetch_ucirepo(id=29)
    computerHardwareDataFrame = pd.DataFrame(computerHardwareData.data.original)
    computerClean = cleaner.clean(computerHardwareDataFrame, [], 'ERP')

    print("FOREST FIRES")
    forestFiresData =  fetch_ucirepo(id=162)
    forestFiresDataFrame = pd.DataFrame(forestFiresData.data.original)
    forestClean = cleaner.clean(forestFiresDataFrame, [], 'area')

    classificationData = {
    'feature1': [100, 150, 200, 250, 150, 200, 250, 300, 100, 150, 200, 250, 150, 200, 250, 300,
                 100, 150, 200, 250, 150, 200, 250, 300, 100, 150, 200, 250, 150, 200],
    'feature2': [500, 600, 700, 800, 600, 700, 800, 900, 500, 600, 700, 800, 600, 700, 800, 900,
                 500, 600, 700, 800, 600, 700, 800, 900, 500, 600, 700, 800, 600, 700],
    'feature3': [900, 1000, 1100, 1200, 1000, 1100, 1200, 1300, 900, 1000, 1100, 1200, 1000, 1100, 1200, 1300,
                 900, 1000, 1100, 1200, 1000, 1100, 1200, 1300, 900, 1000, 1100, 1200, 1000, 1100],
    'feature4': [1300, 1400, 1500, 1600, 1400, 1500, 1600, 1700, 1300, 1400, 1500, 1600, 1400, 1500, 1600, 1700,
                 1300, 1400, 1500, 1600, 1400, 1500, 1600, 1700, 1300, 1400, 1500, 1600, 1400, 1500],
    'class': ['Red', 'Blue', 'Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Green', 'Green',
              'Red', 'Blue', 'Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red', 'Blue', 'Green']
    }
    regressionData = {
    'feature1': [100, 150, 200, 250, 150, 200, 250, 300, 100, 150, 200, 250, 150, 200, 250, 300,
                 100, 150, 200, 250, 150, 200, 250, 300, 100, 150, 200, 250, 150, 200],
    'feature2': [500, 600, 700, 800, 600, 700, 800, 900, 500, 600, 700, 800, 600, 700, 800, 900,
                 500, 600, 700, 800, 600, 700, 800, 900, 500, 600, 700, 800, 600, 700],
    'feature3': [900, 1000, 1100, 1200, 1000, 1100, 1200, 1300, 900, 1000, 1100, 1200, 1000, 1100, 1200, 1300,
                 900, 1000, 1100, 1200, 1000, 1100, 1200, 1300, 900, 1000, 1100, 1200, 1000, 1100],
    'feature4': [1300, 1400, 1500, 1600, 1400, 1500, 1600, 1700, 1300, 1400, 1500, 1600, 1400, 1500, 1600, 1700,
                 1300, 1400, 1500, 1600, 1400, 1500, 1600, 1700, 1300, 1400, 1500, 1600, 1400, 1500],
    'class': [1.24, 3.55, 2.342, 1.235, 4.324, 1.324, 1.54, 2.34, 1.09, 4.242, 5.234, 3.134, 4.242, 1.234, 2.234, 3.86,
              1.234, 1.46, 2.13, 4.76, 3.65, 1.87, 3.45, 3.12, 4.234, 1.68, 3.45, 1.354, 4.231, 5.342]
    }
    # Create DataFrame
    classificationDf = pd.DataFrame(classificationData)
    regressionDf = pd.DataFrame(regressionData)
    hiddenLayers = 1
    neuronsPerLayer = breastCancerTest.shape[0] - 10
    features = breastCancerClean.shape[1] - 1
    classes = 2
    batchSize= 10
    classesList = breastCancerClean['Class'].unique()

    '''
    print("CREATE REGRESSION TEST NETWORK...")
    testRegression = Network.Network(hiddenLayers, neuronsPerLayer, features, 1, "regression", 4)
    regressionTest = Learner.Learner(regressionDf, "regression", "class")
    regressionTest.setNetwork(testRegression)
    #regressionTest.train()
    '''


    print("CREATE CLASSIFICATION TEST LEARNER, ADD TEST NETWORK...")
    test = Network.Network(hiddenLayers, neuronsPerLayer, features, classes, "classification", batchSize, classesList)
    testLearner = Learner.Learner(breastCancerTest, "classification", "Class")
    testLearner.setNetwork(test)
    testLearner.run()
    #testLearner.train()

    print()

main()