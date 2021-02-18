from pathlib import Path
from typing import Dict

import jiwer
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
        self.results = {}
        self.transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemoveWhiteSpace(replace_by_space=False),
        ])
        self.measures = {}
        self.report = {}

    def add_wrapper(self, name: str, wrapper: OCRMethodWrapper) -> None:
        wrapper.set_name(name)
        current_path = EXPERIMENT_PATH.joinpath(name)
        if not current_path.exists():
            current_path.mkdir()
        wrapper.current_path = current_path
        self.wrappers[name] = wrapper

    def execute_wrapper_for_one_image(self, image_path: str, result_path: Path, wrapper: OCRMethodWrapper):
        if result_path.exists():
            return result_path.read_bytes().decode("ISO-8859-1")

        ocr_text = wrapper.get_text(image_path)
        result_path.write_text(ocr_text)
        return ocr_text

    def execute_wrapper_for_one_page(self, comics_page: ComicsPage, wrapper: OCRMethodWrapper) -> None:
        data = self.metadata[comics_page.title]
        for item in data['data']:
            ocr_text = self.execute_wrapper_for_one_image(item['image'],
                                                          wrapper.current_path.joinpath(item['name'] + ".txt"), wrapper)
            self.results[item['name']][wrapper.name] = ocr_text
            self.measures[item['name']][wrapper.name] = self.get_measures_for_texts(self.results[item['name']]["original"], self.results[item['name']][wrapper.name])

    def execute_wrapper_for_all_pages(self, name: str):
        # executor = ProcessPoolExecutor()
        # futures = [executor.submit(self.execute_wrapper_for_one_page, comics_page, self.wrappers[name]) for comics_page
        #            in
        #            get_all_comics_pages()]
        # concurrent.futures.wait(futures)

        for comics_page in get_all_comics_pages():
            self.execute_wrapper_for_one_page(comics_page, self.wrappers[name])

        wer_total = 0
        for entry in self.measures:
            data = self.measures[entry][name]
            wer_total += data["insertions"] + data["deletions"] + data["substitutions"]
        self.report[name] = wer_total / len(self.results)

    def execute_all_wrappers(self) -> None:
        for comics_page in get_all_comics_pages():
            json_path = ocr_experiment_dataset_path.joinpath(comics_page.title + ".json")
            self.metadata[comics_page.title] = jsonpickle.decode(json_path.read_bytes().decode("ISO-8859-1"))

            for item in self.metadata[comics_page.title]['data']:
                self.results[item['name']] = {
                    "original": item['text']
                }
                self.measures[item['name']] = {}

        for name in self.wrappers.keys():
            self.execute_wrapper_for_all_pages(name)

    def get_measures_for_texts(self, expected: str, actual: str):
        if not expected:
            expected = "a"
        if not actual:
            actual = "a"
        return jiwer.compute_measures(
            expected,
            actual,
            truth_transform=self.transformation,
            hypothesis_transform=self.transformation
        )
    
    def save_results(self):
        results_path = EXPERIMENT_PATH.joinpath("results.json")
        results_path.write_bytes(str.encode(jsonpickle.encode(self.results), "ISO-8859-1"))

        measures_path = EXPERIMENT_PATH.joinpath("measures.json")
        measures_path.write_bytes(str.encode(jsonpickle.encode(self.measures), "ISO-8859-1"))

        report_path = EXPERIMENT_PATH.joinpath("report.json")
        report_path.write_bytes(str.encode(jsonpickle.encode(self.report), "ISO-8859-1"))
