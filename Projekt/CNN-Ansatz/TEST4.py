import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb
from wandb.keras import WandbCallback

BATCHSIZE = 8
LR = 0.01 / 16 * BATCHSIZE
FILEPATH = "./Best4.h5"

wandb.init(project="cv_project")

train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    '../DataSetNew/Training',
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    shuffle=True,
    batch_size=BATCHSIZE)

validation_generator = val_datagen.flow_from_directory(
    '../DataSetNew/Validation',
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCHSIZE)

test_generator = test_datagen.flow_from_directory(
    '../DataSetNew/Test',
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCHSIZE)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1.1', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2.1'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3.1'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4.1'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5.1'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(1024, (3, 3), activation='relu', padding='same', name='conv6.1'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(2048, (3, 3), activation='relu', padding='same', name='conv7.1'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(4096, (3, 3), activation='relu', padding='same', name='conv8.1'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8192, (3, 3), activation='relu', padding='same', name='conv9.1'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu', name='fc1', ))
model.add(Dense(16, activation='softmax'))  # Für jedes Label ein output

modelCheckpoint = ModelCheckpoint(FILEPATH, monitor='val_loss', verbose=0, save_best_only=True,
                                  save_weights_only=False, mode='auto', period=1)

reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.25,
                              patience=5, min_lr=0.0005)

earlyStopping = EarlyStopping(patience=15, monitor='val_loss')

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=LR, momentum=0.9),
              metrics=['accuracy'])

model.fit_generator(train_generator,
                    steps_per_epoch=16 * 800 // BATCHSIZE,  # (Num_cat * pics_cat / batchSize)
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=16 * 150 // BATCHSIZE,
                    callbacks=[modelCheckpoint, WandbCallback(), reduceLROnPlateau, earlyStopping])


model.load_weights(FILEPATH, by_name=True)
model.save(os.path.join(wandb.run.dir, "model.h5"))

test_accuracy = model.evaluate_generator(test_generator,
                         steps=16 * 50 // BATCHSIZE)  # (Num_cat * pics_cat / batchSize)

wandb.run.summary["test_accuracy"] = test_accuracy[1]
