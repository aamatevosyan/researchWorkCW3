import base64
import json
import zlib
from pathlib import Path

import cv2
import jsonpickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from SpeechBubblesExperiment import EXPERIMENT_PATH
from speechbubbles.opencv_advanced import opencv_advanced_method
from speechbubbles.opencv_basic import opencv_basic_method
from utils import get_all_comics_pages, compress_numpy_array, get_blank_with_contours, convert_blank_image_to_bw, \
    display_image_in_actual_size

# im = cv.imread('tmp/CYB_BUBBLEGOM_T01_005.jpg.png')
# imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(imgray, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# print(contours)

# configs = get_all_comics_pages()
#
# comics_page = configs[0]
#
# result = opencv_advanced_method(comics_page.href)

from core.SpeechBubblesExperiment import SpeechBubblesExperiment
from methods.speechbubbles.ml import ml_method
from speechbubbles.opencv_advanced import opencv_advanced_method
from speechbubbles.opencv_basic import opencv_basic_method
from utils import prepare_dataset_for_speech_bubbles_experiment

if __name__ == '__main__':
    # prepare_dataset_for_speech_bubbles_experiment()
    #
    # experiment = SpeechBubblesExperiment()
    # experiment.add_method("opencv_basic_method", ml_method)
    # # experiment.add_method("opencv_basic_method", opencv_basic_method)
    # # experiment.add_method("opencv_advanced_method", opencv_advanced_method)
    #
    # experiment.execute_all_methods()

    # json_path = Path("experiments/speechbubbles/opencv_basic_method/CYB_BUBBLEGOM_T01_005.json")
    # arr = jsonpickle.decode(json_path.read_text())
    # res = compress_numpy_array(arr)
    #
    #
    # data2 = base64.b64decode(res)
    # data2 = zlib.decompress(data2)
    #
    # fdata = np.frombuffer(data2, dtype=np.uint8).reshape(arr.shape)

# EXPERIMENT_PATH.
    comics_page = get_all_comics_pages()[0]
    contours = comics_page.get_contours()
    blank_image = get_blank_with_contours(contours, (int(comics_page.height), int(comics_page.width)))
    gray = convert_blank_image_to_bw(blank_image)
    display_image_in_actual_size(gray)

    gray = np.where(gray == 255, 1, gray)

    compare_img_path = "tmp/CYB_BUBBLEGOM_T01_005.jpg.png"
    image = cv2.imread(compare_img_path)
    # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # convert to grayscale
    gray_compare = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray_compare = np.where(gray_compare != 0, 1, gray_compare)

    accuracy = accuracy_score(gray.reshape(-1), gray_compare.reshape(-1))
    print(accuracy)
    print(precision_recall_fscore_support(gray.reshape(-1), gray_compare.reshape(-1), average="binary", labels=[0, 1]))
# print(calc_precision_recall(gray.reshape(-1), gray_compare.reshape(-1)))

# im = cv.imread(comics_page.href)
# imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(imgray, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# print(contours[0].shape)
