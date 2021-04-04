from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import Recall
import os

classifier = Sequential()

image_size = 64
filters = 32
kernel_size = 3
strides = 1
input_shape = (image_size, image_size, 3)
batch_size = 32
pool_size = (3, 3)

classifier.add(Convolution2D(filters=filters, kernel_size=kernel_size, strides=strides, input_shape=input_shape, padding="same", activation='relu'))

classifier.add(MaxPooling2D(pool_size=pool_size))

filters = 64
pool_size = (2, 2)

classifier.add(Convolution2D(filters=filters, kernel_size=kernel_size, strides=strides, input_shape=input_shape, padding="same", activation='relu'))

classifier.add(MaxPooling2D(pool_size=pool_size))

filters = 128

classifier.add(Convolution2D(filters=filters, kernel_size=kernel_size, strides=strides, input_shape=input_shape, padding="same", activation='relu'))

classifier.add(MaxPooling2D(pool_size=pool_size))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dropout(0.5))

classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Recall()])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')

training_set = train_datagen.flow_from_directory(os.path.join(data_dir, 'training_set'),
                                                 target_size=(image_size, image_size),
                                                 batch_size=batch_size,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(os.path.join(data_dir, 'test_set'),
                                            target_size=(image_size, image_size),
                                            batch_size=batch_size,
                                            class_mode='binary')



classifier.fit(training_set,
            steps_per_epoch=8000//batch_size,
            epochs=25,
            validation_data=test_set,
            validation_steps=2000//batch_size)
