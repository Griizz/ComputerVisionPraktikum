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

labels = ["Apple_Green", "Apple_Red", "Banana", "Carambola", "Guava", "Kiwi", "Mango", "Muskmelon", "Orange", "Peach", "Pear", "Persimmon", "Pitaya", "Plum", "Pomegranate", "Tomatoes"]

#Pfadstrings der Trainingsdaten
trStrings = []
for i in range(len(labels)):
    trStrings.append(glob.glob("./DataSet/" + labels[i] + "/Training/*.png"))

#Labels der Trainingsdaten
trLabels = []
for i in range(len(labels)):
    trLabels.append([i] * 800)

#Pfadstrings der Testdaten
testStrings = []
for i in range(len(labels)):
    testStrings.append(glob.glob("./DataSet/" + labels[i] + "/Test/*.png"))

#Labels der Testdaten
testLabels = []
for i in range(len(labels)):
    testLabels.append([i] * 200)


#Einlesen der Bilder
trImgs = []
for i in range(len(trStrings)):
    trImgs.append(imread(trStrings[i]))

testImgs = []
for i in range(len(testStrings)):
    testImgs.append(imread(testStrings[i]))