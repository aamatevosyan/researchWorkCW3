import os
from pathlib import Path
from time import sleep

import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.io import imread
from skimage.transform import resize
from tensorflow.python.keras import models

from utils import convert_bw_image_to_zero_and_ones

model_loaded = False
model_path = 'methods/speechbubbles/0207_e500_std_model_4.h5'
model = models.load_model(model_path)

TMP = Path("tmp")
if not TMP.exists():
    TMP.mkdir()

def ml_method(img_path: str):
    img = imread(img_path)

    height, width, _ = img.shape
    img = resize(img, (768, 512), anti_aliasing=True, preserve_range=True)
    img = np.expand_dims(img, axis=0)
    img = img / 255

    try:
        p = model.predict(img)
        tmp_image_path = TMP.joinpath(os.path.basename(img_path) + ".png")
        arr = img_as_ubyte(p[0, :, :, 0])
        img = arr
        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(tmp_image_path), resized)

        return convert_bw_image_to_zero_and_ones(resized)

    except ValueError as err:
        print(img_path)
        print(err)
        return []
