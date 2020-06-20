import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from tensorflow.keras.applications.vgg16 import VGG16
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D, LeakyReLU
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers as r

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

with open('result_2.log', 'w') as f:
    # Model 1
    f.write('Model 1\n')
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(HEIGHT, WIDTH, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(units=num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    model.summary(print_fn=lambda x: f.write(x+'\n'))

    time_start = time.time()
    history = model.fit(
        train_data,
        train_labels,
        epochs=200,
        validation_data=(test_data, test_labels),
        verbose=2
    )
    time_end = time.time()
    f.write(f'\nTraining time: {time_end - time_start}\n')

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    f.write(f'Test acc: {test_acc}\nTest loss: {test_loss}\n')
    model.save('saved_models/new_1')

    # Model 2
    f.write('Model 2\n')
    model = Sequential([
        Conv2D(32, (8, 8), activation=LeakyReLU(),
               input_shape=(HEIGHT, WIDTH, 3)),
        Conv2D(32, (3, 3), activation=LeakyReLU(), bias_regularizer=r.l2(0.)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (4, 4), activation=LeakyReLU()),
        Conv2D(64, (2, 2), activation=LeakyReLU(), bias_regularizer=r.l2(0.)),
        GlobalMaxPooling2D(),
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dense(units=num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    model.summary(print_fn=lambda x: f.write(x+'\n'))

    time_start = time.time()
    history = model.fit(
        train_data,
        train_labels,
        epochs=200,
        validation_data=(test_data, test_labels),
        verbose=2
    )
    time_end = time.time()
    f.write(f'\nTraining time: {time_end - time_start}\n')

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    f.write(f'Test acc: {test_acc}\nTest loss: {test_loss}\n')
    model.save('saved_models/new_2')
