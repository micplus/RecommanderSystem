import numpy as np
import pandas as pd


def preprocessRatings(path):
    origSet = np.loadtxt(path, delimiter="::")

    users = np.unique(origSet[:, 0])
    items = np.unique(origSet[:, 1])

    trainingSet, validationSet, testSet = divideDataSet(origSet)

    trainingSet = sortDataSet(trainingSet)
    validationSet = sortDataSet(validationSet)
    testSet = sortDataSet(testSet)

    trainingRatingMatrix = getRatingMatrix(trainingSet, users, items)
    validationRatingMatrix = getRatingMatrix(validationSet, users, items)
    testRatingMatrix = getRatingMatrix(testSet, users, items)

    return trainingRatingMatrix, validationRatingMatrix, testRatingMatrix, users, items


def divideDataSet(origSet):
    np.random.shuffle(origSet)  # 打乱数据集，进行划分

    trainLength = int(0.8 * origSet.shape[0])  # 训练集 8
    validationLength = int(0.1 * origSet.shape[0])  # 验证集 1

    trainingSet = origSet[0:trainLength]  # 8
    validationSet = origSet[trainLength:trainLength + validationLength]  # 1
    testSet = origSet[trainLength + validationLength:]  # 1

    return trainingSet, validationSet, testSet


def sortDataSet(dataSet):  # 时间戳从小到大排序
    return dataSet[np.argsort(dataSet[:, 3])]


def getRatingMatrix(dataset, users, items):
    ratingMatrixDF = pd.DataFrame(index=users, columns=items)
    for i in range(dataset.shape[0]):
        ratingMatrixDF.at[dataset[i][0], dataset[i][1]] = dataset[i][2]
    ratingMatrixDF = ratingMatrixDF.fillna(0)
    return ratingMatrixDF.values
