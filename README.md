"""# LaundryLens: A CNN-Based Smart Clothing Classification System

LaundryLens is a Deep Learning project that classifies clothing images into 10 categories using a Convolutional Neural Network (CNN). It uses the Fashion-MNIST dataset and TensorFlow/Keras to train and evaluate an image-classification model.

## Project Objectives

- Build and train a CNN for clothing image classification.
- Preprocess image data for deep learning.
- Evaluate model performance on unseen test images.
- Visualize training progress and compare predicted labels with actual labels.

## Dataset

The project uses **Fashion-MNIST**, which contains 70,000 grayscale images of clothing items:

- 60,000 training images
- 10,000 test images
- Image size: 28 × 28 pixels
- Number of classes: 10

The dataset is loaded directly through `tensorflow.keras.datasets`, so a separate manual download is not required.

### Classes

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Scikit-learn (for the optional classification report and confusion matrix)

## Model Architecture

The CNN uses the following layers:

1. Input layer for 28 × 28 grayscale images
2. Convolution layer with 32 filters and ReLU activation
3. Max-pooling layer
4. Convolution layer with 64 filters and ReLU activation
5. Max-pooling layer
6. Flatten layer
7. Dense layer with 128 units and ReLU activation
8. Dropout layer with a rate of 0.3
9. Output layer with 10 units and softmax activation

The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as a metric.

## Workflow

1. Load the Fashion-MNIST dataset.
2. Visualize example clothing images.
3. Normalize pixel values from the range 0–255 to 0–1.
4. Add a channel dimension for CNN input.
5. Build and train the CNN for 8 epochs, using a batch size of 128 and 10% of the training data for validation.
6. Evaluate the trained model on the separate test dataset.
7. Plot training and validation accuracy and loss.
8. Visualize predictions alongside the actual labels.
9. Save the trained model as `laundrylens_model.keras`.

## Results

The current notebook run achieved:

- **Test accuracy: 90.57%**
- **Test loss: 0.2570**
- **Trainable parameters: 225,034**

These results are from the current run and may vary slightly if the model is retrained.

## How to Run

The easiest way to run this project is with Google Colab.

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload or open `LaundryLens_CNN.ipynb`.
3. Run the notebook cells in order, from top to bottom.
4. Wait for training and evaluation to finish.
5. Review the accuracy/loss graphs and sample predictions.
6. The notebook saves the trained model as `laundrylens_model.keras`. Download it from Colab's Files panel if you want to keep a local copy.

### Required Libraries

Google Colab usually includes the main libraries. If running in a local Python environment, install them with:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Evaluation and Visualizations

The notebook includes:

- Example images from the dataset
- CNN architecture summary
- Training and validation accuracy/loss graphs
- Test-set accuracy and loss
- Sample predictions with actual labels and model confidence

Optional evaluation cells can add a per-class precision, recall, and F1-score report, as well as a confusion matrix using Scikit-learn.

## Limitations

- Fashion-MNIST images are small, grayscale, and 28 × 28 pixels.
- The model is designed for this dataset and is not guaranteed to classify arbitrary photographs of real clothing accurately.
- Confidence scores are model outputs and should not be treated as guarantees of correctness.

## Future Improvements

- Add a simple user interface for selecting test images.
- Compare the CNN with a baseline neural network.
- Experiment with data augmentation and different CNN architectures.
- Analyze class-wise performance and common misclassifications.

## Project Structure

```text
LaundryLens/
├── LaundryLens_CNN.ipynb
├── laundrylens_model.keras   # Created after running the notebook
└── README.md
```

## Conclusion

LaundryLens demonstrates an end-to-end image-classification workflow using a CNN and TensorFlow/Keras. It covers dataset loading, preprocessing, model training, evaluation, visualization, and model saving.

---
**Project:** LaundryLens  
**Type:** Deep Learning / Image Classification  
**Dataset:** Fashion-MNIST
"""
