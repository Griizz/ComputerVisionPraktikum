"""
Aufgabe 5.3
"""

import numpy as np
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from skimage.feature import match_template

#5.1
catG = imread("./catG.png")
catEye = imread("./catEye.png")

#5.2
match_cat = match_template(catG, catEye)
fig, ax = plt.subplots(1,4)

ax[0].imshow(catG, cmap = 'Greys_r')
ax[0].set_title('Original')

ax[1].imshow(catEye, cmap = 'Greys_r')
ax[1].set_title('Template Auge')

ax[2].imshow(match_cat)
ax[2].set_title('Template Match')

#5.3
position_cat = np.unravel_index(np.argmax(match_cat),match_cat.shape)

print(position_cat)

#5.5
wiwally = imread("./whereIsWally1.jpg")
wally = imread("./wally.png")
match_wally = match_template(wiwally,wally)

ax[3].imshow(match_wally[:, :, 0])
ax[3].set_title('Wally Match')

position_wally = np.unravel_index(np.argmax(match_wally),match_wally.shape)
print(position_wally)

"""
5.3.6: 
Das liegt daran, dass Wally auf der Postkarte frontal zu sehen ist und unserem template von der Seite.
Außerdem ist Postkarten Wally auch größer als unser Wally. Deshalb wird er nicht erkannt
"""
