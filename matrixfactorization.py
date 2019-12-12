import numpy as np

from evaluate import evaluate


def train(trainingRatingMatrix, validationRatingMatrix, K, alpha, beta, epochs):
    """
    :param trainingRatingMatrix:训练集评分矩阵
    :param validationRatingMatrix:验证集评分矩阵
    :param K:维度
    :param alpha:学习率
    :param beta:惩罚系数
    :param epochs:迭代上限
    """
    userMatrix, itemMatrix = matrixFactorize(trainingRatingMatrix,
                                             K=K,
                                             alpha=alpha,
                                             beta=beta,
                                             epochs=epochs
                                             )

    predictedRatingMatrix = predict(userMatrix, itemMatrix)

    mseValidation = evaluate(predictedRatingMatrix, validationRatingMatrix)

    return {"userMatrix": userMatrix,
            "itemMatrix": itemMatrix,
            "predictedRatingMatrix": predictedRatingMatrix,
            "mseValidation": mseValidation,
            "K": K,
            "alpha": alpha,
            "beta": beta,
            "epochs": epochs}


def matrixFactorize(origMatrix, K=3, alpha=0.01, beta=0.01, epochs=100000):
    """
    :param origMatrix:待分解矩阵
    :param K:维度
    :param alpha:学习率
    :param beta:惩罚系数
    :param epochs:迭代上限
    """
    # 初始化
    M = origMatrix.shape[0]  # 矩阵大小M*N
    N = origMatrix.shape[1]
    U = np.random.random((M, K))  # 用户矩阵大小M*K
    V = np.random.random((K, N))  # 物品矩阵大小K*N
    loss = 0.  # 损失
    nonzeroNumbers = origMatrix[origMatrix > 0].shape[0]  # 矩阵中有效值数目
    index = origMatrix.nonzero()

    # FunkSVD
    # loss(u_ik,v_kj)=(r_ij-sum(k=1->K, u_ik*v_kj))**2 +
    #                   beta/2*sum(l=1->K, u_ik**2+v_kj**2)
    # 随机梯度下降求解，令e=r_ij-sum(l=1->K, u_ik*v_kj)=r_ij-rHat_ij
    # Gu_loss=-2e_ij*v_kj+beta*u_ik
    # Gv_loss=-2e_ij*u_ik+beta*v_kj
    # 迭代u_il-=Gu_loss, v_lj-=Gv_loss
    for epoch in range(epochs):
        ndx = np.random.choice(nonzeroNumbers)
        # 更新 U,V矩阵
        i = index[0][ndx]
        j = index[1][ndx]
        r = origMatrix[i, j]
        Ui = U[i, :]
        Vj = V[:, j]
        rHat = np.sum(np.dot(Ui, Vj))  # 矩阵化计算
        e = r - rHat
        for k in range(K):  # 更新一行/一列
            Gu = beta * U[i, k] - 2 * e * V[k, j]
            Gv = beta * V[k, j] - 2 * e * U[i, k]
            U[i, k] -= alpha * Gu
            V[k, j] -= alpha * Gv

        # 计算损失
        loss = 0.
        reg = np.sum(Ui ** 2) + np.sum(Vj ** 2)
        e = r - rHat
        loss += e ** 2 + beta * reg / 2

        if epoch % int((epochs / 10)) == 0:
            print("loss={} at epoch {}".format(loss, epoch))
        epoch += 1
    print("loss={} at epoch {}".format(loss, epochs))
    return U, V


def predict(userMatrix, itemMatrix):
    return np.dot(userMatrix, itemMatrix)
