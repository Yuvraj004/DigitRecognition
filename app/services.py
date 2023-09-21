from sklearn.datasets import load_digits
from app.classifier import ClassifierFactory
from app.image_processing import process_image 
import requests
from PIL import Image
from io import BytesIO

class PredictDigitService:
    def __init__(self, repo):
        self.repo = repo

    def handle(self, image_data_uri):
        classifier = self.repo.get()
        if classifier is None:
            digits = load_digits()
            classifier = ClassifierFactory.create_with_fit(
                digits.data, # type: ignore
                digits.target # type: ignore
            )
            self.repo.update(classifier)
        
        # Extract image from URI
        response = requests.get(image_data_uri)
        if response.status_code != 200:
            return 0
        image_data = BytesIO(response.content)
        image = Image.open(image_data)

        # Process image
        x = process_image(image)
        if x is None:
            return 0

        prediction = classifier.predict(x)[0]  # type: ignore
        return prediction
