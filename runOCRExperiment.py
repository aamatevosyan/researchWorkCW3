
# from ocr.EasyOCRWrapper import EasyOCRWrapper
# from ocr.PyteseractWrapper import PyteseractWrapper
#from utils import prepare_dataset_for_ocr_experiment

from core.OCRExperiment import OCRExperiment
from core.ocr.CuneiformWrapper import CuneiformWrapper

if __name__ == '__main__':
    #prepare_dataset_for_ocr_experiment()
    experiment = OCRExperiment()
    # experiment.add_wrapper("pyteseract", PyteseractWrapper())
    # experiment.add_wrapper("easyocr", EasyOCRWrapper())
    experiment.add_wrapper("cuneiform", CuneiformWrapper())
    experiment.execute_all_wrappers()

    print("Done...")