import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input

class Images:
    def __init__(self):
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)

    def load(self, imageA, imageB):
        self.imageA = cv2.imread(imageA)
        self.imageB = cv2.imread(imageB)

    def mask(self, regions: List[Tuple[int, int, int, int]]):
        mask = np.zeros_like(self.imageA[:, :, 0])
        mask = 1 - mask

        for region in regions:
            temp_mask = np.zeros_like(self.imageA[:, :, 0])
            x1, y1, x2, y2 = region
            temp_mask[y1:y2, x1:x2] = 1
            temp_mask = 1 - temp_mask
            mask = mask & temp_mask

        self.imageA = cv2.bitwise_and(self.imageA, self.imageA, mask=mask)
        self.imageB = cv2.bitwise_and(self.imageB, self.imageB, mask=mask)

        cv2.imwrite(os.path.join(self.results_dir, f'imageA_masked_{self.get_datetime()}.jpg'), self.imageA)
        cv2.imwrite(os.path.join(self.results_dir, f'imageB_masked_{self.get_datetime()}.jpg'), self.imageB)

    def pixel_difference(self):
        imageC = self.imageB.copy()

        diff = cv2.absdiff(self.imageA, self.imageB)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        imageC[mask != 0] = [255, 0, 255]

        result = diff.astype(np.uint8)
        score = (np.count_nonzero(result) * 100) / result.size
        print(f"Pixel Difference Score: {score}")

        cv2.imwrite(os.path.join(self.results_dir, f'pixel_difference_{self.get_datetime()}.jpg'), imageC)

        return score

    def structural_similarity_index(self):
        imageA_gray = cv2.cvtColor(self.imageA, cv2.COLOR_BGR2GRAY)
        imageB_gray = cv2.cvtColor(self.imageB, cv2.COLOR_BGR2GRAY)

        score, diff = structural_similarity(imageA_gray, imageB_gray, full=True)
        print(f"Structural Similarity Index (SSIM) Score: {score * 100:.4f}")

        diff = (diff * 255).astype("uint8")
        diff_box = cv2.merge([diff, diff, diff])

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mask = np.zeros(self.imageA.shape, dtype='uint8')
        filled_after = self.imageB.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area > 40:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(self.imageA, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.rectangle(self.imageB, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.rectangle(diff_box, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
                cv2.drawContours(filled_after, [c], 0, (255, 0, 255), -1)

        cv2.imwrite(os.path.join(self.results_dir, f'ssim_imageA_{self.get_datetime()}.jpg'), self.imageA)
        cv2.imwrite(os.path.join(self.results_dir, f'ssim_imageB_{self.get_datetime()}.jpg'), self.imageB)
        cv2.imwrite(os.path.join(self.results_dir, f'ssim_diff_{self.get_datetime()}.jpg'), diff_box)
        cv2.imwrite(os.path.join(self.results_dir, f'ssim_mask_{self.get_datetime()}.jpg'), mask)
        cv2.imwrite(os.path.join(self.results_dir, f'ssim_filled_imageB_{self.get_datetime()}.jpg'), filled_after)

        return score

    def cosine_similarity(self, tolerance=0.000001):
        imageA_resized = cv2.resize(self.imageA, (224, 224))
        imageB_resized = cv2.resize(self.imageB, (224, 224))

        imageA_processed = preprocess_input(imageA_resized)
        imageB_processed = preprocess_input(imageB_resized)

        imageA_features = self.model.predict(np.expand_dims(imageA_processed, axis=0))
        imageB_features = self.model.predict(np.expand_dims(imageB_processed, axis=0))

        score = cosine_similarity(imageA_features, imageB_features)[0][0]
        print(f'Cosine Similarity Score: {score}')

        if abs(score - 1.0) >= tolerance:
            self.pixel_difference()
            self.structural_similarity_index()

        return score

    def get_datetime(self):
        return datetime.now().strftime('%Y%m%d_%H%M%S')