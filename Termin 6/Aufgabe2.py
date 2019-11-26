"""
Aufgabe 6.2
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD

import numpy as np

np.random.seed(123)

from tensorflow import set_random_seed
set_random_seed(123)

def change_labels(label):
    if label == 1:
        return 0

    if label == 4:
        return 1

    if label == 8:
        return 2

td = np.load('./trainingsDatenFarbe2.npz')
vd = np.load('./validierungsDatenFarbe2.npz')

trImages = td['data']
trLabels = td['labels']

vdImages = vd['data']
vdLabels = vd['labels']

trDesk = np.zeros((60, 6))
vdDesk = np.zeros((30, 6))

# 1.2
trDesk[:, :3] = np.mean(trImages, axis=(1, 2))
vdDesk[:, :3] = np.mean(vdImages, axis=(1, 2))

trDesk[:, 3:] = np.std(trImages, axis=(1, 2))
vdDesk[:, 3:] = np.std(vdImages, axis=(1, 2))

for i in range(0, 60):
    trLabels[i] = change_labels(trLabels[i])
for i in range(0, 30):
    vdLabels[i] = change_labels(vdLabels[i])

#Hier drunter ist alt! Weiter mit 2.4

trMatch = [0] * vdDesk.shape[0]
deltaDesk = [0] * trDesk.shape[0]

for i in range(vdDesk.shape[0]):
    x = (trDesk - vdDesk[i])
    for j in range(trDesk.shape[0]):
        deltaDesk[j] = np.sqrt(np.sum((x[j, :]) ** 2))
    n = deltaDesk.index(min(deltaDesk))
    trMatch[i] = td["labels"][n]

tp = sum(list(map(lambda x, y: x == y, trMatch, vd['labels'])))

print("Trefferquote:", tp / len(vdDesk) * 100, "%")



