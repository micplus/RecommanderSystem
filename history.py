def addResultToHistory(history, result):
    history.append(result)
    updateHistory(history)


def updateHistory(history):
    if len(history) > 1:
        newMseValidation = history[len(history) - 1]["mseValidation"]
        if newMseValidation < history[history[0]]["mseValidation"]:
            history[0] = len(history)
    else:
        history[0] = 1


def printBestHistory(history):
    print("\n**BEST: K={}, alpha={}, beta={}, epochs={}"
          .format(history[history[0]]["K"],
                  history[history[0]]["alpha"],
                  history[history[0]]["beta"],
                  history[history[0]]["epochs"]))
