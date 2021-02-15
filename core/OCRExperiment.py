import concurrent
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from typing import Dict

import jsonpickle

from core.comicspage import ComicsPage
from core.ocr.OCRMethodWrapper import OCRMethodWrapper
from core.utils import get_all_comics_pages

EXPERIMENT_PATH = Path("experiments/ocr")
ocr_experiment_dataset_path = EXPERIMENT_PATH.joinpath("dataset")


class OCRExperiment:
    def __init__(self):
        self.wrappers: Dict[str, OCRMethodWrapper] = {}
        self.metadata = {}

    def add_wrapper(self, name: str, wrapper: OCRMethodWrapper) -> None:
        current_path = EXPERIMENT_PATH.joinpath(name)
        if not current_path.exists():
            current_path.mkdir()
        wrapper.current_path = current_path
        self.wrappers[name] = wrapper

    def execute_wrapper_for_one_image(self, image_path: str, result_path: Path, wrapper: OCRMethodWrapper):
        if result_path.exists():
            return

        ocr_text = wrapper.get_text(image_path)
        result_path.write_text(ocr_text)
        return ocr_text

    def execute_wrapper_for_one_page(self, comics_page: ComicsPage, wrapper: OCRMethodWrapper) -> None:
        data = self.metadata[comics_page.title]
        for item in data['data']:
            ocr_text = self.execute_wrapper_for_one_image(item['image'],
                                                          wrapper.current_path.joinpath(item['name'] + ".txt"), wrapper)

    def execute_wrapper_for_all_pages(self, name: str):
        # executor = ProcessPoolExecutor()
        # futures = [executor.submit(self.execute_wrapper_for_one_page, comics_page, self.wrappers[name]) for comics_page
        #            in
        #            get_all_comics_pages()]
        # concurrent.futures.wait(futures)

        for comics_page in get_all_comics_pages():
            self.execute_wrapper_for_one_page(comics_page, self.wrappers[name])

    def execute_all_wrappers(self) -> None:
        for comics_page in get_all_comics_pages():
            json_path = ocr_experiment_dataset_path.joinpath(comics_page.title + ".json")
            self.metadata[comics_page.title] = jsonpickle.decode(json_path.read_bytes().decode("ISO-8859-1"))

        for name in self.wrappers.keys():
            self.execute_wrapper_for_all_pages(name)
