import numpy as np
from classification_model import ClassificationModel


class SobelModel(ClassificationModel):
    def __init__(self, name, threshold):
        super().__init__(name)
        self.threshold = threshold

    def predict(self, image):
        total = np.sum(image)
        value = 1 if total >= self.threshold else 0
        return value

    def get_threshold(self):
        return self.threshold