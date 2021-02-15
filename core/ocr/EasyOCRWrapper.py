from abc import ABC

import easyocr
from ocr.OCRMethodWrapper import OCRMethodWrapper


class EasyOCRWrapper(OCRMethodWrapper, ABC):
    def __init__(self):
        self.reader = easyocr.Reader(['fr', 'en'])

    def get_text(self, img_path: str) -> str:
        return " ".join(self.reader.readtext(img_path, detail=0))
