from abc import ABC
from pathlib import Path

from PIL import Image
import sys

import pyocr
import pyocr.builders

from sys import platform

from pyocr.error import CuneiformError

from core.ocr.OCRMethodWrapper import OCRMethodWrapper

tool = None
lang = None

if platform == "linux":

    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)

    # print(tools)
    # The tools are returned in the recommended order of usage
    tool = tools[0]
    # print("Will use tool '%s'" % (tool.get_name()))
    # Ex: Will use tool 'libtesseract'

    langs = tool.get_available_languages()
    # print("Available languages: %s" % ", ".join(langs))
    lang = langs[2]
    # print("Will use lang '%s'" % (lang))
    # Ex: Will use lang 'fra'
    # Note that languages are NOT sorted in any way. Please refer
    # to the system locale settings for the default language
    # to use.


class CuneiformWrapper(OCRMethodWrapper, ABC):
    def __init__(self):
        self.reader = tool
        self.lang = lang

    def get_text(self, img_path: str) -> str:
        img_path = img_path.replace("\\", "/")
        img_path = f"/home/armen/PycharmProjects/researchWorkCW3/{img_path}"
        print(img_path)
        text = ""
        try:
            text = tool.image_to_string(
                Image.open(img_path),
                lang=self.lang,
            )
        except CuneiformError:
            pass
        print("Done...")

        return text
