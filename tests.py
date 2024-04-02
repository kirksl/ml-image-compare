import unittest
from images import Images

class TestImages(unittest.TestCase):
    def setUp(self):
        self.images = Images()
        print(f'Running test: {self._testMethodName}')

    def test_compare_identical_without_mask(self):
        self.images.load('./images/image1.jpg', './images/image2.jpg')
        self.assertAlmostEqual(self.images.cosine_similarity(0.000001), 1.0, delta=0.000001)

    def test_compare_different_without_mask(self):
        self.images.load('./images/image1.jpg', './images/image3.jpg')
        self.assertAlmostEqual(self.images.cosine_similarity(0.000001), 1.0, delta=0.000001)

    def test_compare_identical_with_mask(self):
        self.images.load('./images/image1.jpg', './images/image3.jpg')
        self.images.mask([(0, 0, 650, 30), (200, 316, 450, 366), (550, 50, 625, 125)])
        self.assertAlmostEqual(self.images.cosine_similarity(0.000001), 1.0, delta=0.000001)

if __name__ == '__main__':
    unittest.main()