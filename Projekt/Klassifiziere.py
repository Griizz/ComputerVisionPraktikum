import numpy as np

def klassifiziere(trDesk, testDesk, trLabels):
    predictions = []
    deltaDesk = [0] * len(trDesk)

    for i in range(len(testDesk)):
        x = (trDesk - testDesk[i])
        for j in range(len(trDesk)):
            deltaDesk[j] = np.sqrt(np.sum((x[j, :]) ** 2))
        n = deltaDesk.index(min(deltaDesk))
        predictions.append(trLabels[n])
    return predictions

trDesk = np.load('trMittelwerte.npy')
testDesk = np.load('testMittelwerte.npy')

trLabels = np.load('trLabels.npy')
testLabels = np.load('testLabels.npy')

predictions = klassifiziere(trDesk, testDesk, trLabels)

evaluatedPredictions = predictions == testLabels
correctPredictions = sum(evaluatedPredictions)
print("Trefferquote:", correctPredictions / len(testLabels) * 100, "%")