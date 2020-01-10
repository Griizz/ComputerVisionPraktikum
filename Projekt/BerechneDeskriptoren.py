"""
Der klassische Ansatz bei unserem Fruits Dataset.

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
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

"""
Gibt bei einer Liste von Bildern entsprechende Masken aus, die nach dem Otsu-Verfahren auf den Saturation-Dimensionen (HSV) maskiert.
"""
def createSMasks(imgs):
    i=0
    masks = []
    for img in imgs:
        imgS = rgb2hsv(img)[:, :, 1]
        masks.append(imgS > threshold_otsu(imgS))
        print(i)
        i +=1
    return masks

def berechneMaskiertenMittelwert(img, mask):
    summe = [0,0,0]
    anzahlRelevanterPixel = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            #if(not(img[x,y,0] == 0 and img[x,y,1] == 0 and img[x,y,2] == 0)):
            if(mask[x,y] != 0):
                anzahlRelevanterPixel +=1
                summe += img[x,y]
    return summe / anzahlRelevanterPixel

def berechneMaskierteMittelwerte(imgs, masks):
    mittelwerte = []
    for i in range(len(imgs)):
        mittelwerte.append(berechneMaskiertenMittelwert(imgs[i], masks[i]))
    return mittelwerte

"""
Erstelle aus einem Bild und einer Maske einen Vektor, der ausschließlich die nach der Maske relevanten Pixel des Bildes in einem Vektor speichert.
Auf diesem Vektor kann danach problemlos np.mean/np.std aufgerufen oder Farbhistogramme berechnet werden
"""
def erstelleVektorRelevanterPixel(img, mask):
    vektor = []
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if(mask[x,y] != 0):
                vektor.append(img[x,y])
    return np.asarray(vektor)

def erstelleVektoren(imgs, masks):
    vektoren = []
    for i in range(len(imgs)):
        vektoren.append(erstelleVektorRelevanterPixel(imgs[i],masks[i]))
        print(i)
    return np.asarray(vektoren)

#Die Test Area:

#testBild1 = imread("./DataSet/Carambola/Training/Carambola_135.png")
#imgS1 = rgb2hsv(testBild1)[:, :, 1]
#mask1 = (imgS1 > threshold_otsu(imgS1))
#testBild2 = imread("./DataSet/Carambola/Training/Carambola_136.png")
#imgS2 = rgb2hsv(testBild2)[:, :, 1]
#mask2 = (imgS2 > threshold_otsu(imgS2))
#testBild1HSV = rgb2hsv(testBild1)
#testBild2HSV = rgb2hsv(testBild2)
#mittelwerteA = berechneMaskierteMittelwerte([testBild1,testBild2], [mask1,mask2])
#vektoren = erstelleVektoren([testBild1,testBild2], [mask1,mask2])
#print(vektoren)
#print(vektoren.shape)
#mittelwerteB =[]
#mittelwerteB.append(np.mean(vektoren[0], axis = 0))
#mittelwerteB.append(np.mean(vektoren[1], axis = 0))
#print(mittelwerteA)
#print(mittelwerteB)

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

print('Einlesen der Bilder:')

#Einlesen der Bilder
i = 0
trImgs = []
for path in trStrings:
    trImgs.append(imread(path))
    print(i)
    i += 1

print('Einlesen der TestBilder:')
i=0
testImgs = []
for path in testStrings:
    testImgs.append(imread(path))
    print(i)
    i += 1

print('Erstelle Masken:')
trMasks = createSMasks(trImgs)
print('Erstelle TestMasken:')
testMasks = createSMasks(testImgs)


#Vektoren berechnen:

print('Erstelle Vektoren:')
trVektoren = erstelleVektoren(trImgs, trMasks)
print('Erstelle TestVektoren:')
testVektoren = erstelleVektoren(testImgs, testMasks)

#Mittelwerte berechnen:

#mit berechneMittelwert:
#trMittelwerte = berechneMaskierteMittelwerte(trImgs, trMasks)
#testMittelwerte = berechneMaskierteMittelwerte(testImgs, testMasks)

#mittels Vektoren und np.mean
#trMittelwerte = []
#for i in range(len(trImgs)):
#    trMittelwerte.append(np.mean(trVektoren[i], axis=0))
#testMittelwerte = []
#for i in range(len(testImgs)):
#    testMittelwerte.append(np.mean(testVektoren[i], axis=0))

#np.save('trMittelwerte',trMittelwerte)
#np.save('testMittelwerte', testMittelwerte)


#Standardabweichungen berechnen:

trStd = []
for i in range(len(trImgs)):
    trStd.append(np.std(trVektoren[i], axis=0))
    print(i)

testStd = []
for i in range(len(testImgs)):
    testStd.append(np.std(testVektoren[i], axis=0))
    print(i)

np.save('trStd', trStd)
np.save('testStd', testStd)

#np.save('trLabels',trLabels)
#np.save('testLabels', testLabels)

