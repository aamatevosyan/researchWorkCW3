from pathlib import Path

import jsonpickle

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
