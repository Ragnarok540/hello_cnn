from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import os

classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')

training_set = train_datagen.flow_from_directory(os.path.join(data_dir, 'training_set'),
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(os.path.join(data_dir, 'test_set'),
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

batch_size = 32

classifier.fit(training_set,
            steps_per_epoch=8000//batch_size,
            epochs=5,
            validation_data=test_set,
            validation_steps=2000//batch_size)

classifier.save('original.cat.dog.model')
