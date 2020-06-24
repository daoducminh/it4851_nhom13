import os

import PIL.Image
from imutils import paths

import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer

WIDTH = 100
HEIGHT = 100

labels = ['mantled_howler', 'patas_monkey', 'bald_uakari', 'japanese_macaque', 'pygmy_marmoset',
          'white_headed_capuchin', 'silvery_marmoset', 'common_squirrel_monkey', 'black_headed_night_monkey', 'nilgiri_langur']
lb = LabelBinarizer()
labels = lb.fit_transform(labels)


def png_to_jpeg(png_file, jpeg_file):
    """
    Convert PNG images to JPEG format
    :param png_file: full path of .png file
    :param jpeg_file: full path of .jpeg file
    """
    im = PIL.Image.open(png_file)
    rgb_im = im.convert('RGB')
    rgb_im.save(jpeg_file, 'JPEG')


def convert_image(image_path):
    if image_path.split('.')[-1].lower() == 'png':
        jpeg_image_path = image_path.replace('.png', '.jpg')
        png_to_jpeg(image_path, jpeg_image_path)
        return jpeg_image_path
    return image_path


def predict(model, image_path):
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (HEIGHT, WIDTH))
    image = np.reshape(image, (HEIGHT, WIDTH, 3))
    data = [image]
    data = np.array(data, dtype='float') / 255.0
    predictions = model.predict(x=data)
    print(predictions)
    a = (predictions > 0.8).astype(int)
    if 1 in a[0]:
        return list(lb.inverse_transform((predictions > 0.7).astype(int)))
    else:
        return []
