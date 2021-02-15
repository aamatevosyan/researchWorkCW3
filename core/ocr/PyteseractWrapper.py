from abc import ABC

import cv2
from pytesseract import pytesseract

from ocr.OCRMethodWrapper import OCRMethodWrapper

pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
custom_config = r'-l eng+fra --psm 6'

class PyteseractWrapper(OCRMethodWrapper, ABC):
    def get_text(self, img_path: str) -> str:
        img_cv = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return pytesseract.image_to_string(img_rgb, config=custom_config)
