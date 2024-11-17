import AlgorithmAccuracy
import Cleaner
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
    
    '''
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
    soybeanData = fetch_ucirepo(id=91)
    soybeanDataFrame = pd.DataFrame(soybeanData.data.original)
    soybeanClean = cleaner.clean(soybeanDataFrame, [], 'class')
    soybeanLearner = Learner.Learner(soybeanClean, "classification", 'class')

    soybeanInfo = classificationAndAccuracyAllLayers(3, soybeanLearner, soybeanClean, "Soybean")

    soybeanLayerFoldClassifications = soybeanInfo[0]  # 3 instances of arrays w/ 10 classification infos
    soybeanTotalClassification = soybeanInfo[1]
    soybeanTotalAccuracyStats = soybeanInfo[2]

    for acc in soybeanTotalAccuracyStats:
        acc.print()

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
    
    print("COMPUTER HARDWARE")
    computerHardwareData = fetch_ucirepo(id=29)
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
    

    print("FOREST FIRES")
    forestFiresData = fetch_ucirepo(id=162)
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

    forestInfo = classificationAndAccuracyAllLayers(3, forestLearner, forestClean, "Forest Fires")

    forestLayerFoldClassifications = forestInfo[0]  # 3 instances of arrays w/ 10 classification infos
    forestTotalClassification = forestInfo[1]
    forestTotalAccuracyStats = forestInfo[2]

    for acc in forestTotalAccuracyStats:
        acc.print()

    # # PLOT LOSS FOR EACH ITERATION OF CROSS-VALIDATION FOR EACH DATASET WITH 0, 1, AND 2 HIDDEN LAYERS IN THE NETWORK
    datasetClassificationInfo = [breastCancerLayerFoldClassifications, glassLayerFoldClassifications,
                                 soybeanLayerFoldClassifications, abaloneLayerFoldClassifications,
                                 computerLayerFoldClassifications, forestLayerFoldClassifications]
    labels = ["Breast Cancer", "Glass", "Soybean", "Abalone", "Computer Hardware", "Forest Fires"]
    numDatasets = len(datasetClassificationInfo)
    numLayers = len(datasetClassificationInfo[0])
    # Number of folds per layer
    numFolds = len(datasetClassificationInfo[0][0])

    # Create a figure with 3 subplots, one for each layer
    fig, axes = plt.subplots(numDatasets, numLayers, figsize=(18, 30), sharey=False)
    for datasetIndex in range(numDatasets):
        for layerIndex in range(numLayers):
            ax = axes[datasetIndex, layerIndex]
            for foldIndex in range(numFolds):
                # Retrieve the loss array for the current layer and fold
                losses = datasetClassificationInfo[datasetIndex][layerIndex][foldIndex].getLoss()

                # Plot the loss values, x-axis being the index within the loss array
                ax.plot(range(len(losses)), losses, label=f'Fold {foldIndex + 1}')

            # Set title and labels for each subplot
            ax.set_title(f'Loss for {labels[datasetIndex]} with {layerIndex} hidden layers')
            ax.set_xlabel('Epochs')
            ax.set_xticks(range(len(losses)))
            if layerIndex == 0:
                ax.set_ylabel('Loss')
            ax.legend(title='Folds', loc='upper right', fontsize='small')  # Legend with folds labeled

    # Adjust layout and display plot
    plt.tight_layout()
    plt.show()

    # datasetAccuracies = [breastCancerTotalAccuracyStats, glassTotalAccuracyStats, soybeanTotalAccuracyStats,
    #                      abaloneTotalAccuracyStats, computerTotalAccuracyStats, forestTotalAccuracyStats]
    # datasetF1Scores = [[] for _ in range(3)]
    # datasetZeroOneLoss = [[] for _ in range(3)]
    #
    # for acc in datasetAccuracies:
    #     for i in range(3):
    #         datasetF1Scores[i].append(acc[i].getF1())
    #         datasetZeroOneLoss[i].append(acc[i].getLoss())
    #
    # # Set width for bars
    # barWidth = 0.35
    #
    # # Create an array for the x-axis
    # x = np.arange(numDatasets)
    #
    # # Create the figure and axis
    # fig, ax = plt.subplots(1, numLayers, figsize=(18, 5))
    #
    # for i in range(numLayers):
    #     # Plot F1 score bars
    #     barsF1 = ax[i].bar(x - barWidth / 2, datasetF1Scores[i], barWidth, label='F1 Score', color='darkcyan')
    #
    #     # Plot 0-1 loss bars
    #     barsLoss = ax[i].bar(x + barWidth / 2, datasetZeroOneLoss[i], barWidth, label='0-1 Loss', color='darkmagenta')
    #
    #     # Set labels, title, and legend for each subplot
    #     ax[i].set_xlabel('Dataset')
    #     ax[i].set_title(f'Accuracy Stats by Dataset for {i} Hidden Layers')
    #     ax[i].set_xticks(x)
    #     ax[i].set_xticklabels(labels, rotation=45, ha='right')
    #     if i == 0:  # Set the y-axis label only on the first subplot to avoid repetition
    #         ax[i].set_ylabel('Scores')
    #     ax[i].legend()
    #
    # # Adjust layout and show plot
    # plt.tight_layout()
    # plt.show()


def classificationAndAccuracyAllLayers(layersRange, learner, cleanData, name):
    layerFoldClassifications = []  # 3 instances of arrays w/ 10 classification infos

    totalClassifications = []  # store classInfo for 0, 1, 2 layers
    totalAccuracies = []  # store accuracy stats for 0, 1, 2 layers

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
            totalClassification.addLoss(np.mean(classInfo.getLoss()))  # add average loss
            totalClassification.addTrueClasses(classInfo.getTrueClasses())  # add true classes

        totalClassifications.append(totalClassification)
        totalAccuracyStats = AlgorithmAccuracy.AlgorithmAccuracy(totalClassification,
                                                                 cleanData.shape[1] - 1,
                                                                 name)
        totalAccuracies.append(totalAccuracyStats)

    return layerFoldClassifications, totalClassifications, totalAccuracies
    '''


main()