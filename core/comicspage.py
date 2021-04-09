import uuid
from pathlib import Path

import cv2
import jsonpickle
import numpy as np

jsonpickle.set_preferred_backend('json')
jsonpickle.set_encoder_options('json', ensure_ascii=False, indent=4)

class ComicsPage:
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @staticmethod
    def open(filename: Path):
        return jsonpickle.decode(filename.read_text(encoding="utf-8"))

    def save(self, filename: Path):
        filename.write_text(jsonpickle.encode(self), encoding="utf-8")

    def get_contours(self) -> []:
        contours = []
        for bubble in self.speech_bubbles:
            contours.append(np.array(bubble["points"]).reshape((-1, 1, 2)).astype(np.int32))
        return contours

    def get_img(self):
        return cv2.imread(self.href)

    def get_contours_bounding_rect_images_and_texts(self):
        im = cv2.imread(self.href)
        contours = self.get_contours()
        texts = []
        for bubble in self.speech_bubbles:
            text = ""
            if 'lines' in bubble:
                for line in bubble['lines']:
                    text += line['text'] + " "
            texts.append(text)

        images = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = im[y:y + h, x:x + w]
            images.append(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        return [images, texts]