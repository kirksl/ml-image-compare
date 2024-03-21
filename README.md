## Purpose

Compare images to determine their similarity using deep learning techniques. This project utilizes TensorFlow, Keras, and ResNet50 to extract features from images and compare them using cosine similarity.

## Setup

To set up the project, follow these steps:

### Requirements

- Python 3.x
- pip

### Installation

1. Clone the repository:

    ```
    git clone https://github.com/kirksl/ml-image-compare.git
    ```

2. Navigate to the project directory:

    ```
    cd ml-image-compare
    ```

3. Install the required packages:

    ```
    pip install -r requirements.txt
    ```

## Execution

1. Run the tests script to compare images:

    ```
    python tests.py
    ```

## Deep Learning Technologies

### TensorFlow

TensorFlow is an open-source machine learning framework developed by Google. It provides a flexible ecosystem for building and deploying machine learning models, including neural networks. TensorFlow allows for efficient computation on both CPUs and GPUs, making it suitable for various tasks in deep learning.

### Keras

Keras is an open-source neural network library written in Python. It provides a high-level API for building and training deep learning models, allowing for rapid prototyping and experimentation. Keras is now integrated into TensorFlow, making it even more accessible to users.

### ResNet50

ResNet50 is a deep convolutional neural network architecture that consists of 50 layers. It is widely used for image classification tasks due to its depth and performance. ResNet50 is pre-trained on the ImageNet dataset and can be fine-tuned or used as a feature extractor for various computer vision tasks.

### Vector and Feature

- **Vector**: In mathematics, a vector is a mathematical object that represents a quantity characterized by both magnitude and direction. In the context of deep learning, a feature vector represents an image's content as a high-dimensional vector, capturing various aspects of the image's visual information.
  
- **Feature**: Features are specific characteristics or patterns extracted from the input data. In image processing, features can include edges, textures, shapes, and object parts, among others. Deep learning models like ResNet50 learn to extract features automatically during the training process.

### Cosine Similarity

Cosine similarity is a measure of similarity between two non-zero vectors in an inner product space. It measures the cosine of the angle between the vectors, indicating how similar or dissimilar they are in terms of their orientations. 

The cosine similarity ranges from -1 to 1:
- **Cosine Similarity = 1**: This indicates that the vectors are pointing in exactly the same direction, meaning they are identical or have the highest possible similarity.
  
- **Cosine Similarity = 0**: This indicates that the vectors are orthogonal (perpendicular) to each other. In the context of comparing feature vectors extracted from images, a cosine similarity of 0 implies that there is no similarity between the images.
  
- **Cosine Similarity = -1**: This indicates that the vectors are pointing in exactly opposite directions, meaning they are as dissimilar as possible.

In the context of comparing images:
- A cosine similarity score close to 1 indicates a high degree of similarity between the images.
- A cosine similarity score close to 0 indicates no similarity between the images.
- A cosine similarity score close to -1 indicates a high degree of dissimilarity between the images.

By comparing feature vectors extracted from different images using cosine similarity, we can quantify their similarity based on the similarity of their underlying visual content. This allows for assessing the similarity or dissimilarity between images in a meaningful way.