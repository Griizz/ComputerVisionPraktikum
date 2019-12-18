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
from skimage.io import imread
from os import walk

#labels = ["Apple_Green", "Apple_Red", "Banana", "Carambola", "Guava", "Kiwi", "Mango", "Muskmelon", "Orange", "Peach", "Pear", "Persimmon", "Pitaya", "Plum", "Pomegranate", "Tomatoes"]
_, labels, _ = walk("./DataSet").__next__()

#Pfadstrings der Trainingsdaten
trStrings = []
for label in labels:
    trStrings += glob.glob("./DataSet/" + label + "/Training/*.png")

#Labels der Trainingsdaten
trLabels = []
for i in range(len(labels)):
    trLabels += [i] * 800

#Pfadstrings der Testdaten
testStrings = []
for label in labels:
    testStrings += glob.glob("./DataSet/" + label + "/Test/*.png")

#Labels der Testdaten
testLabels = []
for i in range(len(labels)):
    testLabels += [i] * 200


#Einlesen der Bilder
trImgs = []
for path in trStrings:
    trImgs.append(imread(path))

testImgs = []
for path in testStrings:
    testImgs.append(imread(path))