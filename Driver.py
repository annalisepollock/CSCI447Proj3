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

    # IMPORT DATA SETS
    print("BREAST CANCER")
    breastCancerData = fetch_ucirepo(id=15)
    breastCancerDataFrame = pd.DataFrame(breastCancerData.data.original)
    breastCancerClean = cleaner.clean(breastCancerDataFrame, ['Sample_code_number'], 'Class')
    breastCancerTest = breastCancerClean.sample(frac=0.5)
    breastCancerLearner = Learner.Learner(breastCancerTest, "classification", "Class")

    breastCancerInfo = classificationAndAccuracyAllLayers(3, breastCancerLearner, breastCancerClean, "Breast Cancer")

    breastCancerLayerFoldClassifications = breastCancerInfo[0]  # 3 instances of arrays w/ 10 classification infos
    breastCancerTotalClassification = breastCancerInfo[1]
    breastCancerTotalAccuracyStats = breastCancerInfo[2]

    print("GLASS")
    glassData = fetch_ucirepo(id=42)
    glassDataFrame = pd.DataFrame(glassData.data.original)
    glassClean = cleaner.clean(glassDataFrame, ['Id_number'], 'Type_of_glass')
    glassLearner = Learner.Learner(glassClean, "classification", 'Type_of_glass')

    glassInfo = classificationAndAccuracyAllLayers(3, glassLearner, glassClean, "Glass")

    glassLayerFoldClassifications = glassInfo[0]  # 3 instances of arrays w/ 10 classification infos
    glassTotalClassification = glassInfo[1]
    glassTotalAccuracyStats = glassInfo[2]

    print("SOYBEAN")
    soybeanData =  fetch_ucirepo(id=91)
    soybeanDataFrame = pd.DataFrame(soybeanData.data.original)
    soybeanClean = cleaner.clean(soybeanDataFrame, [], 'class')
    soybeanLearner = Learner.Learner(soybeanClean, "classification", 'class')

    soybeanInfo = classificationAndAccuracyAllLayers(3, soybeanLearner, soybeanClean, "Soybean")

    soybeanLayerFoldClassifications = soybeanInfo[0]  # 3 instances of arrays w/ 10 classification infos
    soybeanTotalClassification = soybeanInfo[1]
    soybeanTotalAccuracyStats = soybeanInfo[2]

    print("ABALONE")
    abaloneData = fetch_ucirepo(id=1)
    abaloneDataFrame = pd.DataFrame(abaloneData.data.original)
    abaloneClean = cleaner.clean(abaloneDataFrame, [], 'Rings')
    abaloneLearner = Learner.Learner(abaloneClean, "regression", 'Rings')

    abaloneInfo = classificationAndAccuracyAllLayers(3, abaloneLearner, abaloneClean, "Abalone")

    abaloneLayerFoldClassifications = abaloneInfo[0]  # 3 instances of arrays w/ 10 classification infos
    abaloneTotalClassification = abaloneInfo[1]
    abaloneTotalAccuracyStats = abaloneInfo[2]

    print("COMPUTER HARDWARE")
    computerHardwareData =  fetch_ucirepo(id=29)
    computerHardwareDataFrame = pd.DataFrame(computerHardwareData.data.original)
    computerClean = cleaner.clean(computerHardwareDataFrame, [], 'ERP')
    computerLearner = Learner.Learner(computerClean, "regression", 'ERP')

    computerInfo = classificationAndAccuracyAllLayers(3, computerLearner, computerClean, "Computer Hardware")

    computerLayerFoldClassifications = computerInfo[0]  # 3 instances of arrays w/ 10 classification infos
    computerTotalClassification = computerInfo[1]
    computerTotalAccuracyStats = computerInfo[2]

    print("FOREST FIRES")
    forestFiresData =  fetch_ucirepo(id=162)
    forestFiresDataFrame = pd.DataFrame(forestFiresData.data.original)
    forestClean = cleaner.clean(forestFiresDataFrame, [], 'area')
    forestLearner = Learner.Learner(forestClean, "regression", 'area')

    forestInfo = classificationAndAccuracyAllLayers(3, forestLearner, forestClean, "Forest Fires")

    forestLayerFoldClassifications = forestInfo[0]  # 3 instances of arrays w/ 10 classification infos
    forestTotalClassification = forestInfo[1]
    forestTotalAccuracyStats = forestInfo[2]

    # PLOT LOSS DATA
    # losses = [
    #     breastCancerTotalClassification.getLoss(),
    #     glassTotalClassification.getLoss(),
    #     soybeanTotalClassification.getLoss(),
    #     abaloneTotalClassification.getLoss(),
    #     computerTotalClassification.getLoss(),
    #     forestTotalClassification.getLoss()
    # ]
    # labels = [
    #     breastCancerTotalAccuracyStats.getName(),
    #     glassTotalAccuracyStats.getName(),
    #     soybeanTotalAccuracyStats.getName(),
    #     abaloneTotalAccuracyStats.getName(),
    #     computerTotalAccuracyStats.getName(),
    #     forestTotalAccuracyStats.getName()
    # ]
    #
    # # Number of arrays
    # numArrays = len(losses)
    #
    # # Create a figure with a subplot for each array
    # fig, axes = plt.subplots(1, numArrays, figsize=(15, 5))
    #
    # # Plot each array in its own subplot
    # for i, (array, label) in enumerate(zip(losses, labels)):
    #     x = range(len(array))  # x values are the indices
    #     y = array  # y values are the values in the array
    #     axes[i].plot(x, y,linestyle='-')  # Plot values with line and markers
    #     axes[i].set_title("Avg Loss per Fold for " + label)
    #     axes[i].set_xlabel("Index")
    #     if i == 0:
    #         axes[i].set_ylabel("Loss")
    #
    # # Adjust layout and show plot
    # plt.tight_layout()
    # plt.show()

def classificationAndAccuracyAllLayers(layersRange, learner, cleanData, name):
    layerFoldClassifications = []  # 3 instances of arrays w/ 10 classification infos

    totalClassifications = [] # store classInfo for 0, 1, 2 layers
    totalAccuracies = [] # store accuracy stats for 0, 1, 2 layers

    # print("CLEANED DATASET: ")
    for numHiddenLayers in range(layersRange):
        learner.setHiddenLayers(numHiddenLayers)
        classification = learner.run()  # returns array of 10 ClassificationInfos (1 per fold)
        layerFoldClassifications.append(classification)

        totalClassification = ClassificationInfo.ClassificationInfo()

        # Generate total classification across folds for dataset
        for classInfo in classification:
            totalClassification.addTP(classInfo.getTP())  # add true positives
            totalClassification.addTN(classInfo.getTN())  # add true negatives
            totalClassification.addFP(classInfo.getFP())  # add false positives
            totalClassification.addFN(classInfo.getFN())  # add false negatives
            totalClassification.addLoss(np.mean(classInfo.getLoss()))  # add loss
            totalClassification.addTrueClasses(classInfo.getTrueClasses())  # add true classes

        totalClassifications.append(totalClassification)
        totalAccuracyStats = AlgorithmAccuracy.AlgorithmAccuracy(totalClassification,
                                                                         cleanData.shape[1] - 1,
                                                                         name)
        totalAccuracies.append(totalAccuracyStats)

        totalClassification.printAccuracy()
        totalAccuracyStats.print()
    return layerFoldClassifications, totalClassifications, totalAccuracies

main()