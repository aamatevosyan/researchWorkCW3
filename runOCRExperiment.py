from OCRExperiment import OCRExperiment
from ocr.EasyOCRWrapper import EasyOCRWrapper
from ocr.PyteseractWrapper import PyteseractWrapper
from utils import prepare_dataset_for_ocr_experiment

if __name__ == '__main__':
    prepare_dataset_for_ocr_experiment()
    experiment = OCRExperiment()
    experiment.add_wrapper("pyteseract", PyteseractWrapper())
    experiment.add_wrapper("easyocr", EasyOCRWrapper())
    experiment.execute_all_wrappers()

    print("Done...")