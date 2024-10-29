import Network 
import Learner
import pandas as pd
import numpy as np

def main():
    print("CREATE TEST DF...")
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
    neuronsPerLayer = 5
    features = 4 
    classes = 3
    batchSize= 4
    classesList = ["Red", "Blue", "Green"]

    print("CREATE REGRESSION TEST NETWORK...")
    test = Network.Network(hiddenLayers, neuronsPerLayer, features, classes, "classification", batchSize, classesList)
    testRegression = Network.Network(hiddenLayers, neuronsPerLayer, features, 1, "regression", 4)
    regressionTest = Learner.Learner(regressionDf, "regression", "class")
    regressionTest.setNetwork(testRegression)
    regressionTest.train()



    print()

    print("CREATE CLASSIFICATION TEST LEARNER, ADD TEST NETWORK...")
    testLearner = Learner.Learner(classificationDf, "classification", "class")
    testLearner.setNetwork(test)
    testLearner.train()

    print()

main()