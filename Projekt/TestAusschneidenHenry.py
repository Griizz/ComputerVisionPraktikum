"""
Hier ein Beispiel wie man mittels HSV Raum ausschneiden kann. Dabei wird der S mit dem V Anteil multipliziert.
Bei manchen Obst funktioniert es besser als bei anderem: z.B Carambola sehr gut, Plum eher schlecht
"""
from skimage.io import imread, imsave
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu


testBild = imread("./DataSet/Carambola/Training/Carambola_25.png")

imsave("./TestAusschneidenHenry/TestBild.png", testBild)

imgH = rgb2hsv(testBild)[:, :, 0]
imgS = rgb2hsv(testBild)[:, :, 1]
imgV = rgb2hsv(testBild)[:, :, 2]
imgSV = imgS * imgV

imsave("./TestAusschneidenHenry/TestBildH.png", imgH)
imsave("./TestAusschneidenHenry/TestBildS.png", imgS)
imsave("./TestAusschneidenHenry/TestBildV.png", imgV)
imsave("./TestAusschneidenHenry/TestBildSV.png", imgSV)

mask = (imgSV > threshold_otsu(imgSV))

ausgeschnitten  = testBild * mask[ :, :, None]

imsave("./TestAusschneidenHenry/TestAusgeschnitten.png", ausgeschnitten)