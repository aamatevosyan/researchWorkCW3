import cv2
import numpy as np

# Crop image by removing a number of pixels
from utils import get_blank_with_contours, convert_blank_image_to_bw, display_image_in_actual_size, \
    convert_bw_image_to_zero_and_ones


# Comparison function for sorting contours
def get_contour_precedence(contour, cols):
    tolerance_factor = 200
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


# Find all speech bubbles in the given comic page and return a list of their contours
def find_speech_bubbles(image):
    # Convert image to gray scale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Recognizes rectangular/circular bubbles, struggles with dark colored bubbles
    binary = cv2.threshold(image_gray, 235, 255, cv2.THRESH_BINARY)[1]
    # Find contours and document their heirarchy for later
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_map = {}
    final_contour_list = []

    contour_map = filter_contours_by_size(contours)
    contour_map = filter_containing_contours(contour_map, hierarchy)

    # Sort final contour list
    final_contour_list = list(contour_map.values())
    final_contour_list.sort(key=lambda x: get_contour_precedence(x, binary.shape[1]))

    return final_contour_list


def filter_contours_by_size(contours):
    # We could pass this in and update it by reference, but I prefer this sort of 'immutable' handling.
    contour_map = {}

    for i in range(len(contours)):
        # Filter out speech bubble candidates with unreasonable size
        if cv2.contourArea(contours[i]) < 120000 and cv2.contourArea(contours[i]) > 2000:
            # Smooth out contours that were found
            epsilon = 0.0025 * cv2.arcLength(contours[i], True)
            approximated_contour = cv2.approxPolyDP(contours[i], epsilon, True)
            contour_map[i] = approximated_contour

    return contour_map


# Sometimes the contour algorithm identifies entire panels, which can contain speech bubbles already
#  identified causing us to parse them twice via OCR. This method attempts to remove contours that
#  contain other speech bubble candidate contours completely inside of them.
def filter_containing_contours(contourMap, hierarchy):
    # I really wish there was a better way to do this than this O(n^2) removal of all parents in
    #  the heirarchy of a contour, but with the number of contours found this is the only way I can
    #  think of to do this.
    for i in list(contourMap.keys()):
        current_index = i
        while hierarchy[0][current_index][3] > 0:
            if hierarchy[0][current_index][3] in contourMap.keys():
                contourMap.pop(hierarchy[0][current_index][3])
            current_index = hierarchy[0][current_index][3]

    # I'd prefer to handle this 'immutably' like above, but I'd rather not make an unnecessary copy of the dict.
    return contourMap


def opencv_basic_method(img_path: str):
    npimg = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    contours = find_speech_bubbles(img)

    blank_image = get_blank_with_contours(contours, (int(height), int(width)))
    bw = convert_blank_image_to_bw(blank_image)

    # return bw
    return convert_bw_image_to_zero_and_ones(bw)
