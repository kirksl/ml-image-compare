import cv2
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input

class Images:
    def __init__(self):
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    def load(self, imageA, imageB):
        self.imageA = cv2.imread(imageA)
        self.imageB = cv2.imread(imageB)

    def mask(self, regions: List[Tuple[int, int, int, int]]):
        self.mask = np.zeros_like(self.imageA[:, :, 0])
        self.mask = 1 - self.mask

        for region in regions:
            temp_mask = np.zeros_like(self.imageA[:, :, 0])
            x1, y1, x2, y2 = region
            temp_mask[y1:y2, x1:x2] = 1
            temp_mask = 1 - temp_mask
            self.mask = self.mask & temp_mask

        self.imageA = cv2.bitwise_and(self.imageA, self.imageA, mask=self.mask)
        self.imageB = cv2.bitwise_and(self.imageB, self.imageB, mask=self.mask)

    def show(self):
        for image in ('imageA', 'imageB'):
            cv2.imshow(image, getattr(self, image, None))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def resize(self):
        self.imageA = cv2.resize(self.imageA, (224, 224))
        self.imageB = cv2.resize(self.imageB, (224, 224))

    def process(self):
        self.imageA_processed = preprocess_input(self.imageA)
        self.imageB_processed = preprocess_input(self.imageB)

    def extract_features(self):
        self.imageA_features = self.model.predict(np.expand_dims(self.imageA_processed, axis=0))
        self.imageB_features = self.model.predict(np.expand_dims(self.imageB_processed, axis=0))

    def compare(self) -> float:
        score = cosine_similarity(self.imageA_features, self.imageB_features)[0][0]
        return score