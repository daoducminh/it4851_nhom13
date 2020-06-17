import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow.keras as keras
from imutils import paths
from lxml import etree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential

num_classes = 10
WIDTH = 100
HEIGHT = 100
train_image_folders = 'data/monkeys/training/n{}/'
test_image_folders = 'data/monkeys/validation/n{}/'

train_data = []
train_labels = []

test_data = []
test_labels = []

with open('error.log', 'w') as fp:
    for i in range(10):
        train_image_paths = list(
            paths.list_images(train_image_folders.format(i)))

        for image_path in train_image_paths:
            try:
                image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (HEIGHT, WIDTH))
                image = np.reshape(image, (HEIGHT, WIDTH, 3))

                train_data.append(image)
                train_labels.append(i)
            except Exception:
                fp.write(image_path+'\n')
        # =========================================================================
        test_image_paths = list(
            paths.list_images(test_image_folders.format(i)))

        for image_path in test_image_paths:
            try:
                image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (HEIGHT, WIDTH))
                image = np.reshape(image, (HEIGHT, WIDTH, 3))

                test_data.append(image)
                test_labels.append(i)
            except Exception:
                fp.write(image_path+'\n')

train_data = np.array(train_data, dtype='float') / 255.0
train_labels = np.array(train_labels)

test_data = np.array(test_data, dtype='float') / 255.0
test_labels = np.array(test_labels)

lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.transform(test_labels)

with open('result.log', 'w') as f:
    # Model 1
    f.write('Model 1\n')
    model = Sequential()

    model.add(Conv2D(40, (3, 3), activation='relu',
                     input_shape=(HEIGHT, WIDTH, 3)))
    model.add(Conv2D(80, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(80, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(80, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(80, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    time_start = time.time()
    history = model.fit(
        train_data,
        train_labels,
        epochs=20,
        validation_data=(test_data, test_labels),
        verbose=2
    )
    time_end = time.time()
    f.write(f'Training time: {time_end - time_start}\n')

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    f.write(f'Test acc: {test_acc}\nTest loss: {test_loss}\n')
    model.save('saved_models/trained_cnn1')

    # Model 2
    f.write('Model 2\n')
    model = Sequential()

    model.add(Conv2D(20, (3, 3), activation='relu',
                     input_shape=(HEIGHT, WIDTH, 3)))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(40, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    time_start = time.time()
    history = model.fit(
        train_data,
        train_labels,
        epochs=20,
        validation_data=(test_data, test_labels),
        verbose=2
    )
    time_end = time.time()
    f.write(f'Training time: {time_end - time_start}\n')

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    f.write(f'Test acc: {test_acc}\nTest loss: {test_loss}\n')
    model.save('saved_models/trained_cnn2')

    # Model 3
    f.write('Model 3\n')
    model = Sequential()

    model.add(Conv2D(20, (3, 3), activation='relu',
                     input_shape=(HEIGHT, WIDTH, 3)))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(Conv2D(40, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    time_start = time.time()
    history = model.fit(
        train_data,
        train_labels,
        epochs=20,
        validation_data=(test_data, test_labels),
        verbose=2
    )
    time_end = time.time()
    f.write(f'Training time: {time_end - time_start}\n')

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    f.write(f'Test acc: {test_acc}\nTest loss: {test_loss}\n')
    model.save('saved_models/trained_cnn3')

    # Model 4
    f.write('Model 4\n')
    model = Sequential()

    model.add(Conv2D(20, (3, 3), activation='relu',
                     input_shape=(HEIGHT, WIDTH, 3)))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(40, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    time_start = time.time()
    history = model.fit(
        train_data,
        train_labels,
        epochs=20,
        validation_data=(test_data, test_labels),
        verbose=2
    )
    time_end = time.time()
    f.write(f'Training time: {time_end - time_start}\n')

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    f.write(f'Test acc: {test_acc}\nTest loss: {test_loss}\n')
    model.save('saved_models/trained_cnn4')

    # Model 5
    f.write('Model 5\n')
    model = Sequential()

    model.add(Conv2D(40, (3, 3), activation='relu',
                     input_shape=(HEIGHT, WIDTH, 3)))
    model.add(Conv2D(80, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(80, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(80, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(80, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(80, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    time_start = time.time()
    history = model.fit(
        train_data,
        train_labels,
        epochs=20,
        validation_data=(test_data, test_labels),
        verbose=2
    )
    time_end = time.time()
    f.write(f'Training time: {time_end - time_start}\n')

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    f.write(f'Test acc: {test_acc}\nTest loss: {test_loss}\n')
    model.save('saved_models/trained_cnn5')

    # Model 6
    f.write('Model 6\n')
    model = Sequential()

    model.add(Conv2D(80, (3, 3), activation='relu',
                     input_shape=(HEIGHT, WIDTH, 3)))
    model.add(Conv2D(160, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(160, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(160, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(160, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(80, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    time_start = time.time()
    history = model.fit(
        train_data,
        train_labels,
        epochs=20,
        validation_data=(test_data, test_labels),
        verbose=2
    )
    time_end = time.time()
    f.write(f'Training time: {time_end - time_start}\n')

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    f.write(f'Test acc: {test_acc}\nTest loss: {test_loss}\n')
    model.save('saved_models/trained_cnn6')

    # Model 7
    f.write('Model 7\n')
    model = Sequential()

    model.add(Conv2D(20, (3, 3), activation='relu',
                     input_shape=(HEIGHT, WIDTH, 3)))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(80, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    time_start = time.time()
    history = model.fit(
        train_data,
        train_labels,
        epochs=20,
        validation_data=(test_data, test_labels),
        verbose=2
    )
    time_end = time.time()
    f.write(f'Training time: {time_end - time_start}\n')

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    f.write(f'Test acc: {test_acc}\nTest loss: {test_loss}\n')
    model.save('saved_models/trained_cnn7')

    # Model 8
    f.write('Model 8\n')
    model = Sequential()

    model.add(Conv2D(80, (3, 3), activation='relu',
                     input_shape=(HEIGHT, WIDTH, 3)))
    model.add(Conv2D(160, (3, 3), activation='relu'))
    model.add(Conv2D(160, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(80, (3, 3), activation='relu'))
    model.add(Conv2D(80, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(Conv2D(40, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    time_start = time.time()
    history = model.fit(
        train_data,
        train_labels,
        epochs=20,
        validation_data=(test_data, test_labels),
        verbose=2
    )
    time_end = time.time()
    f.write(f'Training time: {time_end - time_start}\n')

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    f.write(f'Test acc: {test_acc}\nTest loss: {test_loss}\n')
    model.save('saved_models/trained_cnn8')

    # Model 9
    f.write('Model 9\n')
    model = Sequential()

    model.add(Conv2D(80, (3, 3), activation='relu',
                     input_shape=(HEIGHT, WIDTH, 3)))
    model.add(Conv2D(160, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(160, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(80, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(80, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    time_start = time.time()
    history = model.fit(
        train_data,
        train_labels,
        epochs=20,
        validation_data=(test_data, test_labels),
        verbose=2
    )
    time_end = time.time()
    f.write(f'Training time: {time_end - time_start}\n')

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    f.write(f'Test acc: {test_acc}\nTest loss: {test_loss}\n')
    model.save('saved_models/trained_cnn9')

    # Model 10
    f.write('Model 10\n')
    model = Sequential()

    model.add(Conv2D(320, (3, 3), activation='relu',
                     input_shape=(HEIGHT, WIDTH, 3)))
    model.add(Conv2D(160, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(80, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(40, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    time_start = time.time()
    history = model.fit(
        train_data,
        train_labels,
        epochs=20,
        validation_data=(test_data, test_labels),
        verbose=2
    )
    time_end = time.time()
    f.write(f'Training time: {time_end - time_start}\n')

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    f.write(f'Test acc: {test_acc}\nTest loss: {test_loss}\n')
    model.save('saved_models/trained_cnn10')
