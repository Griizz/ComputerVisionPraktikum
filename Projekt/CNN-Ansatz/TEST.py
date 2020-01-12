import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb
from wandb.keras import WandbCallback

wandb.init(project="cv_project")

train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    '../DataSetNew/Training',
    target_size=(256, 256),
    batch_size=32,)

validation_generator = val_datagen.flow_from_directory(
    '../DataSetNew/Validation',
    target_size=(256, 256),
    batch_size=32,)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1', input_shape=(256, 256, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (8, 8), activation='relu', padding='same', name='conv3'))
model.add(Conv2D(32, (8, 8), activation='relu', padding='same', name='conv4'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (8, 8), activation='relu', padding='same', name='conv5'))
model.add(Conv2D(32, (8, 8), activation='relu', padding='same', name='conv6'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu', name='fc1', ))
model.add(Dense(16, activation='softmax'))  # Für jedes Label ein output

modelCheckpoint = ModelCheckpoint("./Best.h5", monitor='val_loss', verbose=0, save_best_only=True,
                                  save_weights_only=False, mode='auto', period=1)

reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.0,
                              patience=5, min_lr=0.001)

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, momentum=0.9),
              metrics=['accuracy'])

model.fit_generator(train_generator,
                    steps_per_epoch=800,  # 16 * 800 / 32 (Num_cat * pics_cat / batchSize)
                    epochs=50,
                    validation_data=validation_generator,
                    validation_steps=75,  # 16 * 150 / 32 (Num_cat * pics_cat / batchSize)
                    callbacks=[modelCheckpoint, WandbCallback(), reduceLROnPlateau])


model.load_weights("./Best.h5", by_name=True)
model.save(os.path.join(wandb.run.dir, "model.h5"))
