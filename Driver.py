import Network 
import Learner
import pandas as pd

def main():

    print("CREATE TEST DF...")
    data = {
    'feature1': [1, 2, 3, 4, 2, 3, 4, 5, 1, 2, 3, 4],
    'feature2': [5, 6, 7, 8, 6, 7, 8, 9, 5, 6, 7, 8],
    'feature3': [9, 10, 11, 12, 10, 11, 12, 13, 9, 10, 11, 12],
    'feature4': [13, 14, 15, 16, 14, 15, 16, 17, 13, 14, 15, 16],
    'class': ['Red', 'Blue', 'Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red']
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    hiddenLayers = 3
    neuronsPerLayer = 5
    features = 4 
    classes = 3
    classesList = ["Red", "Blue", "Green"]
    batchSize = 2

    print("CREATE TEST NETWORK...")
    test = Network.Network(hiddenLayers, neuronsPerLayer, features, classes, classesList, "classification", batchSize)
    test.printNetwork()
    print()

    print("CREATE TEST LEARNER, ADD TEST NETWORK...")
    testLearner = Learner.Learner(df, "classification", "class")
    testLearner.setNetwork(test)

    print("RUN ONE FORWARD PASS...")
    testData = df.sample(n=2)
    testClasses = testData["class"].to_numpy()
    testData = testData.drop(columns=["class"])
    print("Test Classes: ")
    print(testClasses)
    testLearner.setTestClass(testClasses)
    print("Test Data: ")
    print(testData)
    print()
    output = test.forwardPass(testData)
    print("Classified As:")
    print(output)

    print()

main()