"""
Aufgabe 7.1
"""

import numpy as np

from tensorflow import random

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping, ModelCheckpoint

# um die Gewichte immer gleichzufaellig zu initialisieren
np.random.seed(123)
random.set_seed(123)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Erweitern des Trainingssatzes
X_train = np.concatenate((X_train, np.flip(X_train, axis=2)), axis=0)
y_train = np.concatenate((y_train, y_train), axis=0)

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (8, 8), activation='relu', padding='same', name='conv3'))
model.add(Conv2D(32, (8, 8), activation='relu', padding='same', name='conv4'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu', name='fc1', ))
model.add(Dense(10, activation='softmax'))  # Für jedes Label ein output

earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
modelCheckpoint = ModelCheckpoint("./Best.h5", monitor='val_loss', verbose=0, save_best_only=True,
                                  save_weights_only=False, mode='auto', period=1)

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, momentum=0.9),
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, nb_epoch=20, validation_split=0.2,
          callbacks=[earlyStopping, modelCheckpoint], verbose=1)

model.load_weights("./Best.h5", by_name=True)
score = model.evaluate(X_test, y_test, verbose=1)

print(score)
# nach 20 Epochen acc= 76%
