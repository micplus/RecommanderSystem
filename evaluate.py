import numpy as np


def evaluate(predictedRatingMatrix, testRatingMatrix):
    testValues = testRatingMatrix[testRatingMatrix > 0]

    tempMatrix = testRatingMatrix.copy()    # 用于去除0值，仅对比两矩阵中均有值的部分
    tempMatrix[tempMatrix == 0] = -999
    tempMatrix[tempMatrix > 0] = 0

    compareRatingMatrix = predictedRatingMatrix + tempMatrix
    compareValues = compareRatingMatrix[compareRatingMatrix >= 0]

    mse = np.sum((testValues - compareValues) ** 2) / testValues.shape[0]
    print("MSE={}".format(mse))
    return mse


def evaluateBestFromHistory(history, testRatingMatrix):
    return evaluate(history[history[0]]["predictedRatingMatrix"], testRatingMatrix)
