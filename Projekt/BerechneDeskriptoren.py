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
from skimage.feature import hog
ANZAHLPIXEL = 320 * 258

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
erwartet vektor als eingabe
"""
def erstelle_1d_histos_gewichtet(imgs,b):
    i = 0
    _hist_vektor = []
    for img in imgs:
        histR = np.histogram(img[:,0], bins=b, range=(0, 256))[0]
        histG = np.histogram(img[:,1], bins=b, range=(0, 256))[0]
        histB = np.histogram(img[:,2], bins=b, range=(0, 256))[0]
        histR = histR * (ANZAHLPIXEL/len(img))
        histG = histG * (ANZAHLPIXEL/len(img))
        histB = histB * (ANZAHLPIXEL/len(img))
        _hist_vektor.append(np.hstack((histR, histG, histB)))
        print(i)
        i += 1
    return np.asarray(_hist_vektor)


def erstelle_1d_histo(imgs):
    _hist_vektor = []
    for img in imgs:
        histR = np.histogram(img[:,0], bins=8, range=(0, 256))[0]
        histG = np.histogram(img[:,1], bins=8, range=(0, 256))[0]
        histB = np.histogram(img[:,2], bins=8, range=(0, 256))[0]
        _hist_vektor.append(np.dstack((histR, histG, histB)))
    return np.asarray(_hist_vektor)


"""
Hier wir die Vektorfkt benutzt um die maskierten Stellen zu entfernen
"""
def erstelle_3d_histos_gewichtet(imgs,b):
    _hist3d = []
    for img in imgs:

        _hist3d.append(np.histogramdd(img, bins = [b,b,b], range=((0,256),(0,256),(0,256)))[0]*ANZAHLPIXEL/len(img))

    return np.asarray(_hist3d)


"""
Hier wir die Vektorfkt benutzt um die maskierten Stellen zu entfernen, b ist anzahl bins
"""
def erstelle_3d_histos_mit_vektor(imgs, b):
    _hist3d = []
    for img in imgs:

        _hist3d.append(np.histogramdd(img, bins = [b,b,b], range=((0,256),(0,256),(0,256)))[0])

    return np.asarray(_hist3d)


"""
Hier ist bei der range die 0 nicht berücksichtigt weil bei den input Bildern nur die Pixel den Wert 0 haben die durch die Maske entfernt wurden.
"""
def erstelle_3d_histo(imgs):
    _hist3d = []
    for i in range(len(imgs)):

        _hist3d.append(np.histogramdd(imgs[i].reshape((imgs[i].shape[0]*imgs[i].shape[1],3)), bins = [8,8,8], range=((1,256),(1,256),(1,256)))[0])

    return np.asarray(_hist3d)

def berechneKanten(imgs,masks):
    Bildkanten = []
    for i in range(len(imgs)):
        gray = cv.cvtColor(imgs[i], cv.COLOR_BGR2GRAY)
        img_sobel = sobel(gray, mask=None)
        _img = img_sobel*masks[i]
        Bildkanten.append(_img)
        
    return np.asarray(Bildkanten)
"""
Berechnet 3d HOG und erstellt daraus dann ein 3d histo, hog hat Winkelinformationen deshalb ist die Range 0-360
"""
def berechne_HOG(imgs):
    hogs = []
    for i in range(len(imgs)):
        imgS = rgb2hsv(imgs[i])[:, :, 1]
        mask = (imgS > threshold_otsu(imgS))

        # Calculate gradient
        gx = cv.Sobel(imgs[i], cv.CV_32F, 1, 0, ksize=1)
        gy = cv.Sobel(imgs[i], cv.CV_32F, 0, 1, ksize=1)
        mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)

        vektor = []
        for x in range(imgs[i].shape[0]):
            for y in range(imgs[i].shape[1]):
                if (mask[x, y] != 0):
                    vektor.append(angle[x, y, :])
        vek = np.asarray(vektor)
        _hist = np.histogramdd(vek, bins=[8, 8, 8], range=((0, 360), (0, 360), (0, 360)))[0]
        hogs.append(_hist)
    return hogs

tr_hog = berechne_HOG(trImgs)
np.save('tr_hog',tr_hog)
test_hog = berechne_HOG(testImgs)
np.save('test_hog',test_hog)

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


def berechne_hog(imgs,masks):
    _hogs = []
    for i in range(len(imgs)):
        maske = masks[i]
        img_mask = imgs[i]*maske[:,:, None]
        fd, img_hog = hog(img_mask, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
        _hogs.append(img_hog)
    return _hogs


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
#histos1d = erstelle_1d_histos_gewichtet(vektoren)
#print(histos1d)
#histos3d = erstelle_3d_histos_mit_vektor(vektoren)
#print(histos3d)


#labels = ["Apple_Green", "Apple_Red", "Banana", "Carambola", "Guava", "Kiwi", "Mango", "Muskmelon", "Orange", "Peach", "Pear", "Persimmon", "Pitaya", "Plum", "Pomegranate", "Tomatoes"]
#_, labels, _ = walk("./DataSet").__next__()

#Pfadstrings der Trainingsdaten
#trStrings = []
#for label in labels:
#    trStrings += glob.glob("./DataSet/" + label + "/Training/*.png")

#Haelfte der Pfadstrings
#trStrings = []
#for label in labels:
#    for i in range(400):
#        trStrings.append("./DataSet/" + label + "/Training/"+label+"_"+str(i)+".png")

#Labels der Trainingsdaten
#trLabels = []
#for i in range(len(labels)):
#    trLabels += [i] * 800

#Haelfte der Labels
#trLabels = []
#for i in range(len(labels)):
#    trLabels += [i] * 400

#Pfadstrings der Testdaten
#testStrings = []
#for label in labels:
#    testStrings += glob.glob("./DataSet/" + label + "/Test/*.png")

#Labels der Testdaten
#testLabels = []
#for i in range(len(labels)):
#    testLabels += [i] * 200

#np.save('trLabels400',trLabels)
#np.save('testLabels400', testLabels)

#print('Einlesen der Bilder:')

#Einlesen der Bilder
#i = 0
#trImgs = []
#for path in trStrings:
#    trImgs.append(imread(path))
#    print(i)
#    i += 1

#print('Einlesen der TestBilder:')
#i=0
#testImgs = []
#for path in testStrings:
#    testImgs.append(imread(path))
#    print(i)
#    i += 1

#print('Erstelle Masken:')
#trMasks = createSMasks(trImgs)
#print('Erstelle TestMasken:')
#testMasks = createSMasks(testImgs)


#HOGs berechnen
#tr_hogs = berechne_hog(trImgs,trMasks)
#test_hogs = berechne_hog(testImgs, testMasks)

#np.save('trHOG_400', tr_hogs)
#np.save('testHOG_400', test_hogs)


#Vektoren berechnen:

#print('Erstelle Vektoren:')
#trVektoren = erstelleVektoren(trImgs, trMasks)
#print('Erstelle TestVektoren:')
#testVektoren = erstelleVektoren(testImgs, testMasks)

#np.save('trVektoren', trVektoren)
#np.save('testVektoren', testVektoren)


#Vektoren laden:
trVektoren = np.load('trVektoren.npy',allow_pickle= True)
testVektoren = np.load('testVektoren.npy', allow_pickle= True)

#Mittelwerte berechnen:

#nicht maskiert:
#trMittelwerte = []
#for i in range(len(trImgs)):
#    trMittelwerte.append(np.mean(trImgs[i], axis = (0,1)))
#testMittelwerte = []
#for i in range(len(testImgs)):
#    testMittelwerte.append(np.mean(testImgs[i], axis = (0,1)))

#np.save('trMittelwerteDumm', trMittelwerte)
#np.save('testMittelwerteDumm',testMittelwerte)

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

#np.save('trMittelwerte400',trMittelwerte)
#np.save('testMittelwerte400', testMittelwerte)

#Standardabweichungen berechnen:

#trStd = []
#for i in range(len(trImgs)):
#    trStd.append(np.std(trVektoren[i], axis=0))
#    print(i)

#testStd = []
#for i in range(len(testImgs)):
#    testStd.append(np.std(testVektoren[i], axis=0))
#    print(i)

#np.save('trStd', trStd)
#np.save('testStd', testStd)


#1D Histos gewichtet berechnen:
#print("Erstelle 1D Histos Training:")
#tr1DHistosGewichtet = erstelle_1d_histos_gewichtet(trVektoren,32)
#print("Erstelle 1D Histos Test:")
#test1DHistosGewichtet = erstelle_1d_histos_gewichtet(testVektoren,32)

#np.save('tr1DHistosGewichtet32bins', tr1DHistosGewichtet)
#np.save('test1DHistosGewichtet32bins', test1DHistosGewichtet)


#3D Histos:
#print("Erstelle 3D Histos Training:")
#tr3DHistos = erstelle_3d_histos_mit_vektor(trVektoren,4)
#print("Erstelle 3D Histos Test:")
#test3DHistos = erstelle_3d_histos_mit_vektor(testVektoren,4)

#np.save('tr3DHistos4bins', tr3DHistos)
#np.save('test3DHistos4bins', test3DHistos)

#3D Histos gewichtet:
print("Erstelle 3D Histos Training:")
tr3DHistosGewichtet = erstelle_3d_histos_gewichtet(trVektoren,32)
print("Erstelle 3D Histos Test:")
test3DHistosGewichtet = erstelle_3d_histos_gewichtet(testVektoren,32)

np.save('tr3DHistosGewichtet32bins', tr3DHistosGewichtet)
np.save('test3DHistosGewichtet32bins', test3DHistosGewichtet)

#np.save('trLabels',trLabels)
#np.save('testLabels', testLabels)

