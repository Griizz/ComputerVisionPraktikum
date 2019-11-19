"""
Aufgabe 5.2
"""

import numpy as np
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from skimage.filters import sobel_h, sobel_v, gaussian

img1 = imread("./catG.png")
img_vertical = sobel_v(img1, mask = None)
img_horizontal = sobel_h(img1, mask = None)
img_Sum = np.sqrt(((img_vertical)**2)+((img_horizontal)**2)) #berechnet man so den Gradienten?

imgN = imread("./catGNoisy.png")
imgN_vertical = sobel_v(imgN, mask = None)
imgN_horizontal = sobel_h(imgN, mask = None)
imgN_Sum = np.sqrt(((imgN_vertical)**2)+((imgN_horizontal)**2))
"""
5.2.2: 
Beim vorher gegaussten Bild sind die Linien viel breiter.Das kann daran liegen, dass starke Unterschiede zwischen 
benachbarten Pixeln, in einer geringeren intensität, auf eine größere Fläche verteilt werden  
"""

img2 = gaussian(imgN,5)
img2_vertical = sobel_v(img2, mask = None)
img2_horizontal = sobel_h(img2, mask = None)
img2_Sum = np.sqrt(((img2_vertical)**2)+((img2_horizontal)**2))


fig, ax = plt.subplots(1,3)

ax[0].imshow(img_Sum, cmap = 'Greys_r')
ax[0].set_title('CatG')

ax[1].imshow(imgN_Sum, cmap = 'Greys_r')
ax[1].set_title('CatG Noisy')

ax[2].imshow(img2_Sum, cmap = 'Greys_r')
ax[2].set_title('CatG Noisy gauss')
