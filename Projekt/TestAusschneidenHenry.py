"""
Hier ein Beispiel wie man mittels HSV Raum ausschneiden kann. Dabei wird der S mit dem V Anteil multipliziert.
Bei manchen Obst funktioniert es besser als bei anderem: z.B Carambola sehr gut, Plum eher schlecht
"""
from skimage.io import imread, imsave
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from keras.utils import np_utils
import cv2 as cv

testBild = imread("./DataSet/Carambola/Training/Carambola_136.png")

imsave("./TestAusschneidenHenry/TestBild.png", testBild)

imgH = rgb2hsv(testBild)[:, :, 0]
imgS = rgb2hsv(testBild)[:, :, 1]
imgV = rgb2hsv(testBild)[:, :, 2]
imgSV = 0.75* imgS + 0.25 * imgV

#imgS_8bit = (imgS * 255).astype("uint8")

imsave("./TestAusschneidenHenry/TestBildH.png", imgH)
imsave("./TestAusschneidenHenry/TestBildS.png", imgS)
imsave("./TestAusschneidenHenry/TestBildV.png", imgV)
imsave("./TestAusschneidenHenry/TestBildSV.png", imgSV)

#th_mean_grey = cv.adaptiveThreshold(imgS_8bit,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)


mask = (imgS > threshold_otsu(imgS))


ausgeschnitten  = testBild * mask[ :, :, None]

imsave("./TestAusschneidenHenry/TestAusgeschnitten.png", ausgeschnitten)
imsave("./TestAusschneidenHenry/TestMask.png", mask)

#cv.imshow('Bild 2', mask)



#cv.waitKey(0)
#cv.destroyAllWindows()

