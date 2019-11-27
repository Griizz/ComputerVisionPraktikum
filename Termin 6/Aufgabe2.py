"""
Aufgabe 6.2
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD
from tensorflow import random

import numpy as np

# make results reproducible
np.random.seed(123)
random.set_seed(123)

def change_labels(label):
    '''
    A function mapping values from teh set {1,4,8} to the set {0,1,2}
    :param label: the current label from {1,4,8}
    :return: returns the representing Label from {0,1,2}
    '''
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

trLabels = np_utils.to_categorical(trLabels, 3)
vdLabels = np_utils.to_categorical(vdLabels, 3)

model = Sequential()

model.add(Dense(8, activation='relu', name='fc1', input_shape=(6,)))
model.add(Dense(8, activation='relu', name='fc2'))
model.add(Dense(3, activation='softmax', name='result'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.000005, momentum=0.9), 
              metrics=['accuracy'])



model.fit(trDesk, trLabels, batch_size=1, nb_epoch=500, verbose=1)

score = model.evaluate(vdDesk, vdLabels, verbose=1)

print("Loss:", score[0], "\nAcc:", score[1])
#nach 500 Epochen acc = 56.67%

