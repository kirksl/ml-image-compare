import unittest
from images import Images

class TestImages(unittest.TestCase):
    def setUp(self):
        self.images = Images()

    def test_compare_identical_without_mask(self):
        self.images.load('./images/image1.jpg', './images/image2.jpg')
        self.images.resize()
        self.images.process()
        self.images.extract_features()
        self.assertGreaterEqual(self.images.compare(), 1.0)

    def test_compare_identical_with_mask(self):
        self.images.load('./images/image1.jpg', './images/image3.jpg')
        self.images.mask([(0, 0, 650, 30), (200, 316, 450, 366), (550, 50, 625, 125)])
        self.images.show()
        self.images.resize()
        self.images.process()
        self.images.extract_features()
        self.assertGreater(self.images.compare(), .9)

if __name__ == '__main__':
    unittest.main()