import csv
import csv
import os
from pathlib import Path
from typing import Callable

import cv2
import jsonpickle
import numpy as np

from core.comicspage import ComicsPage
from speechbubbles.opencv_basic import find_speech_bubbles
from utils import get_all_comics_pages, PAGES_PATH, compress_numpy_array, get_numpy_array_from_json, \
    get_precision_recall_f1_score, display_image_in_actual_size

EXPERIMENT_PATH = Path("experiments/speechbubbles")


class SpeechBubblesExperiment:
    def __init__(self):
        self.methods = {}

    def add_method(self, name: str, func: Callable[[str], list]) -> None:
        current_path = EXPERIMENT_PATH.joinpath(name)
        if not current_path.exists():
            current_path.mkdir()

        self.methods[name] = func

    def execute_method_for_one_page(self, name: str, comics_page: ComicsPage) -> None:
        func = self.methods[name]

        json_path = EXPERIMENT_PATH.joinpath(name, comics_page.title + ".json")
        img_path = str(PAGES_PATH.joinpath(comics_page.title + ".jpg"))

        if json_path.exists():
            return

        print(json_path)

        result = func(img_path)
        compressed_data = compress_numpy_array(result)

        json_path.write_text(jsonpickle.encode({
            "data": compressed_data,
            "shape": result.shape
        }))

    def get_bounding_box_for_one_page(self, name: str, comics_page: ComicsPage) -> None:

        # txt file

        txt_path = EXPERIMENT_PATH.joinpath(name, comics_page.title + ".txt")

        # if txt_path.exists():
        #     return

        json_path = EXPERIMENT_PATH.joinpath(name, comics_page.title + ".json")
        json_str = json_path.read_text()
        result = get_numpy_array_from_json(json_str)
        binary = np.where(result != 0, 255, result)
        backtorgb = cv2.cvtColor(binary,cv2.COLOR_GRAY2RGB)

        # contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = find_speech_bubbles(backtorgb)
        lines = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            lines.append(f"balloon {0.3} {x} {y} {w} {h}")

        contours = comics_page.get_contours()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # lines.append(f"balloon {0.3} {x} {y} {w} {h}")
        
        display_image_in_actual_size(backtorgb)
        
        txt_path.write_text("\n".join(lines))

    def get_bounding_box_for_all_pages(self, name: str):
        # executor = ProcessPoolExecutor()
        # futures = [executor.submit(self.execute_method_for_one_page, name, comics_page) for comics_page in
        #            get_all_comics_pages()]
        # concurrent.futures.wait(futures)

        for comics_page in get_all_comics_pages():
            self.get_bounding_box_for_one_page(name, comics_page)

    def get_bounding_box_for_all_methods(self) -> None:
        for name in self.methods.keys():
            self.get_bounding_box_for_all_pages(name)

    def execute_method_for_all_pages(self, name: str):
        # executor = ProcessPoolExecutor()
        # futures = [executor.submit(self.execute_method_for_one_page, name, comics_page) for comics_page in
        #            get_all_comics_pages()]
        # concurrent.futures.wait(futures)

        for comics_page in get_all_comics_pages():
            self.execute_method_for_one_page(name, comics_page)

    def execute_all_methods(self) -> None:
        for name in self.methods.keys():
            self.execute_method_for_all_pages(name)

    def set_dataset(self, dataset, json_path):
        filename, extension = os.path.splitext(os.path.basename(str(json_path.as_posix())))
        json_str = json_path.read_text()
        fdata = get_numpy_array_from_json(json_str)
        dataset[filename] = fdata.ravel()

    def get_score(self, csv_data, comics_page, dataset):
        entry = {"title": comics_page.title}
        for name in self.methods.keys():
            json_path = EXPERIMENT_PATH.joinpath(name, comics_page.title + ".json")
            json_str = json_path.read_text()
            fdata = get_numpy_array_from_json(json_str).ravel()
            precision, recall, f1_score, _ = get_precision_recall_f1_score(dataset[comics_page.title], fdata)

            entry[name + "_precision"] = precision
            entry[name + "_recall"] = recall
            entry[name + "_f1_score"] = f1_score
        print(entry)
        csv_data.append(entry)

    def get_scores_for_all_methods(self):

        csv_data = []
        fieldnames = ["title"]
        for name in self.methods.keys():
            fieldnames.append(name + "_precision")
            fieldnames.append(name + "_recall")
            fieldnames.append(name + "_f1_score")
        dataset = {}

        for json_path in EXPERIMENT_PATH.joinpath("dataset").glob("*.json"):
            self.set_dataset(dataset, json_path)

        # print(dataset)

        for comics_page in get_all_comics_pages():
            self.get_score(csv_data, comics_page, dataset)

        # futures = [executor.submit(self.get_score, csv_data, comics_page, dataset) for comics_page in
        #            get_all_comics_pages()]
        # concurrent.futures.wait(futures)

        print(csv_data)

        with open(EXPERIMENT_PATH.joinpath("scores.csv"), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(csv_data)
