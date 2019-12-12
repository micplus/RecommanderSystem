import contentbasedrecommander as cb
from evaluate import evaluate
from preprocessmovies import preprocessMovies
from preprocessratings import preprocessRatings


def run():
    trainingRatingMatrix, validationRatingMatrix, testRatingMatrix, users, items = \
        preprocessRatings("./data/ml-1m/ratings.dat")
    factorMatrix= preprocessMovies("./data/ml-1m/movies.dat", items)
    predictedRatingMatrix = cb.predict(cb.train(trainingRatingMatrix, factorMatrix), factorMatrix)
    mseTest = evaluate(predictedRatingMatrix, testRatingMatrix)
    return {"predictedRatingMatrix": predictedRatingMatrix,
            "mseTest": mseTest}
