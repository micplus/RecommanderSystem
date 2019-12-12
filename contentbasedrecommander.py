import numpy as np


def train(userRatingMatrix, itemFactorMatrix):
    userFactorMatrix = getUserFactorMatrix(userRatingMatrix, itemFactorMatrix)
    return userFactorMatrix


def predict(userFactorMatrix, itemFactorMatrix):
    predictedMatrix = userFactorMatrix.dot(itemFactorMatrix.T)
    for i in range(predictedMatrix.shape[0]):  # user
        userFactorMatrixSumI = np.sum(userFactorMatrix[i] ** 2)
        for j in range(predictedMatrix.shape[1]):  # item
            predictedMatrix[i, j] /= np.sqrt(userFactorMatrixSumI * np.sum(itemFactorMatrix[j] ** 2))
    return normalize(predictedMatrix, 0, 5)


def getUserFactorMatrix(userRatingMatrix, itemFactorMatrix):  # 用户对特征的偏好权重
    return userRatingMatrix.dot(itemFactorMatrix)


def normalize(X, lower, upper):
    xMin = np.min(X)
    xMax = np.max(X)
    k = (upper - lower) / (xMax - xMin)
    return lower + k * (X - xMin)
