import numpy as np
import pandas as pd
import re


def preprocessMovies(path, items):
    origSet = pd.read_csv(path, sep="::",
                          header=None, names=["MovieID", "Title", "Genres"], engine="python")
    movieIdDF = origSet["MovieID"]

    origSet = origSet[movieIdDF.isin(items)]
    origSet = np.array(origSet)
    origSet = pd.DataFrame(origSet)
    origSet.columns = ["MovieID", "Title", "Genres"]

    factorMatrix= getFactorMatrix(origSet, "([^|]*)")

    return factorMatrix


def getFactorMatrix(origSet, pattern):
    regex = re.compile(pattern, re.I)
    for i in range(len(origSet["Genres"])):
        groups = regex.findall(origSet["Genres"][i])
        for j in range(len(groups)):
            if groups[j] != "":
                origSet.at[i, groups[j]] = 1

    factorMatrix = np.array(origSet.iloc[:, 3:].fillna(0))
    return factorMatrix
