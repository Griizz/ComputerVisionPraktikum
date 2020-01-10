import numpy as np
from os import walk

def klassifiziereNN(trDesk, testDesk, trLabels):
    predictions = []
    deltaDesk = [0] * len(trDesk)

    for i in range(len(testDesk)):
        x = (trDesk - testDesk[i])
        for j in range(len(trDesk)):
            deltaDesk[j] = np.sqrt(np.sum((x[j, :]) ** 2))
        n = deltaDesk.index(min(deltaDesk))
        predictions.append(trLabels[n])
    return predictions

def ConfusionMatrix(prediction, validation, labelCount):
    '''

    :param prediction: the 1-Dimensional Array of predicted Labels
    :param validation: the 1-Dimensional Array of actual Labels
    :param labelCount: the numer of labels
    :return: a labelCount x labelCount ConfusionMatrix
    '''

    matrix = np.zeros((labelCount, labelCount), dtype=int)

    for i in range(len(prediction)):
        matrix[validation[i], prediction[i]] += 1

    return matrix

_, labels, _ = walk("./DataSet").__next__()

#Wähle Mittelwert als Deskriptor
trDesk = np.load('trMittelwerte.npy')
testDesk = np.load('testMittelwerte.npy')

trLabels = np.load('trLabels.npy')
testLabels = np.load('testLabels.npy')

predictions = klassifiziereNN(trDesk, testDesk, trLabels)


#Berechnung der Trefferquote
evaluatedPredictions = predictions == testLabels
#for i in range(len(predictions)):
#    if predictions[i] != testLabels[i]:
#        print(labels[testLabels[i]] + " seen as " + labels[predictions[i]])
correctPredictions = sum(evaluatedPredictions)
print("Trefferquote:", correctPredictions / len(testLabels) * 100, "%")

#Berechnung der Confusion Matrix
cMatrix = ConfusionMatrix(predictions, testLabels, 16)
print(cMatrix)