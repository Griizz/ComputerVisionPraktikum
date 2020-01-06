"""
Der CNN Ansatz bei unserem Fruits Dataset.

Labels:
0  - Apple Green
1  - Apple Red
2  - Banana
3  - Carambola
4  - Guava
5  - Kiwi
6  - Mango
7  - Muskmelon
8  - Orange
9  - Peach
10 - Pear
11 - Persimmon
12 - Pitaya
13 - Plum
14 - Pomegranate
15 - Tomato
"""

import glob
import numpy as np
from keras.utils import np_utils
from skimage.io import imread
from os import walk

def Load(PATH = "../DataSet/"):
    '''

    :param PATH:
    :return:
    '''

    _, labels, _ = walk(PATH).__next__()

    #Pfadstrings der Trainingsdaten
    trStrings = []
    for label in labels:
        trStrings += glob.glob(PATH + label + "/Training/*.png")

    #Labels der Trainingsdaten
    trLabels = []
    for i in range(len(labels)):
        trLabels += [i] * 800

    trLabels = np_utils.to_categorical(trLabels)


    #Pfadstrings der Testdaten
    testStrings = []
    for label in labels:
        testStrings += glob.glob("./DataSet/" + label + "/Test/*.png")

    #Labels der Testdaten
    testLabels = []
    for i in range(len(labels)):
        testLabels += [i] * 200

    testLabels = np_utils.to_categorical(testLabels)


    #Einlesen der Bilder
    trImgs = []
    for path in trStrings:
        trImgs.append(imread(path))

    testImgs = []
    for path in testStrings:
        testImgs.append(imread(path))

    return (np.array(trImgs), trLabels, np.array(testImgs), testLabels)
