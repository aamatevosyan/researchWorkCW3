from pathlib import Path
from typing import Any, Callable, List

import cv2
import jsonpickle

from core.comicspage import ComicsPage
import pprint

from core.svgparser import get_all_configs

pp = pprint.PrettyPrinter(indent=4)

EXPERIMENT_PATH = Path("experiments/speechbubbles")

class SpeechBubblesExperiment:
    def __init__(self):
        self.methods = []

    def add_method(self, method_name: str, method):
        current_path = EXPERIMENT_PATH.joinpath(method_name)
        if not current_path.exists():
            current_path.mkdir()

        tmp = {
            "name": method_name,
            "method": method
        }

        self.methods.append(tmp)
        return tmp

    @staticmethod
    def execute_method_for_all(method_dict: dict):
        for comicsPage in get_all_configs():
            SpeechBubblesExperiment.execute_method(method_dict, comicsPage)

    @staticmethod
    def execute_method(method_dict: dict, data: ComicsPage):
        json_path = EXPERIMENT_PATH.joinpath(method_dict["name"], data.title + ".json")
        preprocess_img_path = EXPERIMENT_PATH.joinpath(method_dict["name"], data.title + "_preprocessed" + ".jpg")
        method = method_dict["method"]

        result = method(data.href, str(preprocess_img_path))
        json_path.write_text(jsonpickle.encode(result))