import numpy as np
from os import walk
from sklearn.neighbors import KNeighborsClassifier
from sys import maxsize

#"""
#k-Nearest Neighbour Klassifikator
#"""
#def klassifiziereKNN(trDesk, testDesk, trLabels, k):
#    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
#    knn.fit(trDesk, trLabels)
#    return knn.predict(testDesk)

"""
k-Nearest Neighbour Klassifikator
"""
def klassifiziereKNN(trDesk, testDesk, trLabels, k):
    predictions = []
    deltaDesk = [0] * len(trDesk)

    for i in range(len(testDesk)):
        x = (trDesk - testDesk[i])
        for j in range(len(trDesk)):
            deltaDesk[j] = np.sqrt(np.sum((x[j, :]) ** 2))
        labels = np.zeros(16)
        for x in range(k):
            index = np.argmin(deltaDesk)
            label = trLabels[index]
            labels[label] +=1
            deltaDesk[index] = maxsize
        predictions.append(np.argmax(labels))
    return predictions

"""
Nearest Neighbour Klassifikator
"""
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

#Mittelwert Dumm als Deskriptor
trDesk = np.load('trMittelwerteDumm.npy')
testDesk = np.load('testMittelwerteDumm.npy')

#Wähle Mittelwert als Deskriptor
#trDesk = np.load('trMittelwerte.npy')
#testDesk = np.load('testMittelwerte.npy')

#Wähle Mittelwert und Standardabweichung als Deskriptor
#trMittelwerte = np.load('trMittelwerte.npy')
#testMittelwerte = np.load('testMittelwerte.npy')
#trStd = np.load('trStd.npy')
#testStd = np.load('testStd.npy')
#trDesk = np.hstack((trMittelwerte,trStd))
#testDesk = np.hstack((testMittelwerte, testStd))

#Wähle 1D Histos gewichtet als Deskriptor:
#trDesk = np.load('tr1DHistosGewichtet.npy')
#testDesk = np.load('test1DHistosGewichtet.npy')

#Wähle 3D Histos als Deskriptor:
#trDesk = np.load('tr3DHistosGewichtet.npy')
#testDesk = np.load('test3DHistosGewichtet.npy')

trLabels = np.load('trLabels.npy')
testLabels = np.load('testLabels.npy')

#Nearest Neighbour
predictions = klassifiziereNN(trDesk, testDesk, trLabels)

#Berechnung der Trefferquote
evaluatedPredictions = predictions == testLabels
correctPredictions = sum(evaluatedPredictions)
print("Trefferquote:", correctPredictions / len(testLabels) * 100, "%")

#Berechnung der Confusion Matrix
cMatrix = ConfusionMatrix(predictions, testLabels, 16)
print(cMatrix)

#k-NearestNeighbour
#for i in range(10):
#    k = i +1
#    predictions = klassifiziereKNN(trDesk, testDesk, trLabels, k)
#    evaluatedPredictions = predictions == testLabels
#    correctPredictions = sum(evaluatedPredictions)
#    print("Trefferquote k = "+str(k)+":", correctPredictions / len(testLabels) * 100, "%")

#    # Berechnung der Confusion Matrix
#    cMatrix = ConfusionMatrix(predictions, testLabels, 16)
#    print(cMatrix)