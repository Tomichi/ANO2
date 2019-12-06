from classification_model import ClassificationModel


class CNNModel(ClassificationModel):
    def __init__(self, name, model):
        super().__init__(name)
        self.model = model

    def get_model(self):
        return self.model

    def predict(self, image):
        prediction = self.model.predict(image)
        value = 1 if prediction > 0.5 else 0
        return value
