import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import measure
from tensorflow.python.keras import models
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf

model_path = 'methods/speechbubbles/0207_e500_std_model_4.h5'
model = models.load_model(model_path)


def show_bubbles(im_path: str):
    image = cv2.imread(im_path)

    # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # create a binary thresholded image
    _, binary = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    # show it

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width, _ = image.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(image, [cnt], 0, 255, -1)

    cv2.imwrite(im_path, image)
    return contours


def ml_method(img_path: str, preprocess_img_path: str):
    img = imread(img_path)
    img = resize(img, (768, 512), anti_aliasing=True, preserve_range=True)
    img = np.expand_dims(img, axis=0)
    img = img / 255

    try:
        p = model.predict(img)
        imsave(fname=preprocess_img_path, arr=p[0, :, :, 0])
        print(preprocess_img_path)

        return show_bubbles(preprocess_img_path)
    except ValueError as err:
        print(err)
        return []
