import matrixfactorization as mf
import history as ht
import evaluate as el
from preprocessratings import preprocessRatings


def run():
    trainingRatingMatrix, validationRatingMatrix, testRatingMatrix, users, items = \
        preprocessRatings("./data/ml-1m/ratings.dat")
    history = [-1]  # 保存历史记录，首项表示最佳结果在列表中的位置
    ht.addResultToHistory(history,
                          mf.train(trainingRatingMatrix, validationRatingMatrix,
                                   K=10, alpha=0.005, beta=0.1, epochs=1000000))
    ht.printBestHistory(history)
    mseTest = el.evaluateBestFromHistory(history, testRatingMatrix)
    return {"predictedRatingMatrix": history[history[0]]["predictedRatingMatrix"],
            "mseTest": mseTest}
