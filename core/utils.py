import base64
import os
import zlib
from pathlib import Path
from typing import List

import cv2
import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
import xmltodict
from sklearn.metrics import precision_recall_fscore_support

from Rect import get_overlap_clusters, Point, Rect
from core.comicspage import ComicsPage

jsonpickle.set_preferred_backend('json')
jsonpickle.set_encoder_options('json', ensure_ascii=False, indent=4)

ORIGIN_PATH = Path("assets/origin")
PAGES_PATH = Path("assets/pages")
GT_PATH = Path("assets/gt")


def compress_numpy_array(arr):
    data = zlib.compress(arr)
    data = base64.b64encode(data)
    return data


def decompress_numpy_array(data, shape):
    data2 = base64.b64decode(data)
    data2 = zlib.decompress(data2)
    fdata = np.frombuffer(data2, dtype=np.uint8).reshape(shape)

    return fdata


def get_numpy_array_from_json(json_str: str):
    obj = jsonpickle.decode(json_str)
    return decompress_numpy_array(obj["data"], obj["shape"])


def convert_svg_to_json(src: Path) -> None:
    filename, extension = os.path.splitext(os.path.basename(str(src)))
    destination = ORIGIN_PATH.joinpath(filename + ".json")

    if destination.exists():
        return

    res = xmltodict.parse(src.read_text(encoding="utf-8"))

    root = res["svg"]
    title = root["title"]
    contents = root["svg"]

    width = contents[0]["image"]["@width"]
    height = contents[0]["image"]["@height"]
    href = str(PAGES_PATH.joinpath(filename + ".jpg").as_posix())

    bubbles = {}

    for content in contents:
        if content["@class"] == "Balloon":
            polygon = content["polygon"]

            for pol in polygon:
                bubbles[pol["@id"]] = {
                    "points": [list(map(lambda x: int(x), el.split(","))) for el in pol["@points"].split(" ")],
                    "metadata": {
                        "border_style": pol["metadata"]["@borderStyle"],
                        "tail_tip": pol["metadata"]["@tailTip"],
                        "bounding_box": pol["metadata"]["@boundingBox"],
                        "tail_direction": pol["metadata"]["@tailDirection"],
                        "rank": pol["metadata"]["@rank"]
                    },
                    "lines": [],
                }
        elif content["@class"] == "Line":
            polygon = content["polygon"]
            for pol in polygon:
                if "@idBalloon" in pol["metadata"]:
                    key = pol["metadata"]["@idBalloon"]
                    bubbles[key]["lines"].append({
                        "points": [list(map(lambda x: int(x), el.split(","))) for el in pol["@points"].split(" ")],
                        "text_type": pol["metadata"]["@textType"],
                        "text": pol["metadata"]["#text"]
                    })

    speech_bubbles = []

    for key, value in bubbles.items():
        bubble = value
        bubble["id"] = key
        speech_bubbles.append(bubble)

    result = {
        "title": title,
        "width": width,
        "height": height,
        "href": href,
        "speech_bubbles": speech_bubbles
    }

    ComicsPage(result).save(destination)


def get_all_comics_pages() -> List[ComicsPage]:
    return comics_pages


def display_image_in_actual_size(img):
    dpi = 80

    if len(img.shape) == 2:
        height, width = img.shape
    else:
        height, width, depth = img.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(img, cmap='gray')

    plt.show()


def get_blank_with_contours(contours, shape):
    height, width = shape
    blank_image = np.zeros(shape=[height, width, 3], dtype=np.uint8)

    for cnt in contours:
        cv2.fillPoly(blank_image, [cnt], [255, 255, 255])

    return blank_image


def convert_blank_image_to_bw(img):
    # convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray


def convert_bw_image_to_zero_and_ones(img):
    return np.where(img != 0, 1, img)


def get_precision_recall_f1_score(orig, pred):
    return precision_recall_fscore_support(orig, pred, average="binary", labels=[0, 1])


def prepare_dataset_for_speech_bubbles_experiment():
    for comics_page in comics_pages:
        json_path = speech_bubbles_experiment_dataset_path.joinpath(comics_page.title + ".json")
        if json_path.exists():
            return

        txt_path = speech_bubbles_experiment_dataset_path.joinpath(comics_page.title + ".txt")

        if txt_path.exists():
            return

        contours = comics_page.get_contours()
        blank_image = get_blank_with_contours(contours, (int(comics_page.height), int(comics_page.width)))
        bw = convert_blank_image_to_bw(blank_image)
        result = convert_bw_image_to_zero_and_ones(bw)

        lines = []
        rectangles = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            rectangles.append(Rect(Point(x, y), Point(x + w, y + h)))

        contours = get_overlap_clusters(rectangles)
        for cnt in contours:
            lines.append(f"balloon {cnt.left} {cnt.top} {cnt.right - cnt.left} {cnt.top - cnt.bottom}")

        txt_path.write_text("\n".join(lines))

        compressed_data = compress_numpy_array(result)

        json_path.write_text(jsonpickle.encode({
            "data": compressed_data,
            "shape": result.shape
        }))

def prepare_dataset_for_ocr_experiment():
    for comics_page in comics_pages:
        json_path = ocr_experiment_dataset_path.joinpath(comics_page.title + ".json")
        data = {"title": comics_page.title, "data": []}

        # if json_path.exists():
        #     return

        [images, texts] = comics_page.get_contours_bounding_rect_images_and_texts()

        for i in range(len(images)):
            cv2.imwrite(str(ocr_experiment_dataset_path.joinpath(comics_page.title + f"_{i}.jpg")), images[i])
            ocr_experiment_dataset_path.joinpath(comics_page.title + f"_{i}.txt").write_text(texts[i], encoding="utf-8")

            data['data'].append({
                "name": comics_page.title + f"_{i}",
                "image": str(ocr_experiment_dataset_path.joinpath(comics_page.title + f"_{i}.jpg")),
                "text": texts[i]
            })

        json_path.write_text(jsonpickle.encode(data))

for filename in GT_PATH.glob("*.svg"):
    convert_svg_to_json(filename)

comics_pages: List[ComicsPage] = []
for file_name in ORIGIN_PATH.glob("*.json"):
    comics_pages.append(ComicsPage.open(file_name))

speech_bubbles_experiment_path = Path("experiments/speechbubbles")
speech_bubbles_experiment_dataset_path = speech_bubbles_experiment_path.joinpath("dataset")

ocr_experiment_path = Path("experiments/ocr")
ocr_experiment_dataset_path = ocr_experiment_path.joinpath("dataset")

if not speech_bubbles_experiment_dataset_path.exists():
    speech_bubbles_experiment_dataset_path.mkdir()

if not ocr_experiment_dataset_path.exists():
    ocr_experiment_dataset_path.mkdir()
