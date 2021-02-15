from abc import ABC, abstractmethod


class OCRMethodWrapper(ABC):
    @abstractmethod
    def get_text(self, img_path: str) -> str:
        ...