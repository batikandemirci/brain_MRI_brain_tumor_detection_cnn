# Brain MRI Tumor Detection using CNN

This project aims to detect brain tumors from MRI images using a Convolutional Neural Network (CNN). The model is trained to classify MRI images as either "tumor" or "no tumor" based on features learned during training. We use a labeled dataset of MRI images with two classes and leverage data preprocessing, CNN layers, and evaluation metrics to achieve accurate detection.

## Dataset

The dataset used in this project is [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection), which contains:
- **98 images** of MRIs without tumors (`no` class)
- **155 images** of MRIs with tumors (`yes` class)

Each image is grayscale and has been resized to 128x128 pixels for model training.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV
- Scikit-Learn
- Matplotlib
- Numpy

You can install all required libraries using:
```bash
pip install tensorflow opencv-python scikit-learn matplotlib numpy
```

## Project Overview

1. **Data Loading and Preprocessing**  
   - Load images from the 'no' and 'yes' folders and assign class labels: `0` for no tumor and `1` for tumor.
   - Convert images to grayscale, resize them to 128x128 pixels, and normalize pixel values to the range [0, 1].

2. **Data Splitting**  
   - Split the dataset into **80% training** and **20% testing** sets.
   - Reshape the images to include a channel dimension (128x128x1), as required by the CNN model.

3. **Model Architecture**  
   - Build a Convolutional Neural Network (CNN) using the following layers:
      - **Conv2D + ReLU**: Extracts feature maps from images.
      - **MaxPooling2D**: Reduces the spatial dimensions of feature maps.
      - **Flatten**: Converts 2D feature maps to a 1D vector for the dense layer.
      - **Dense (ReLU)**: Fully connected layer for learning complex patterns.
      - **Dropout**: Reduces overfitting by randomly deactivating neurons.
      - **Dense (Sigmoid)**: Outputs a probability for binary classification.
   - Compile the model using `binary_crossentropy` as the loss function and `Adam` as the optimizer.

4. **Training**  
   - Train the model with `epochs=10` and a `batch_size=16` to optimize the parameters.
   - Use the test set as validation data to monitor performance.

5. **Evaluation**  
   - Evaluate the model's accuracy on the test set.
   - Plot the training and validation accuracy and loss to analyze model performance.

6. **Confusion Matrix and Classification Report**  
   - Generate a confusion matrix to evaluate true positives, false positives, true negatives, and false negatives.
   - Create a classification report for metrics like precision, recall, and F1-score to assess model performance on each class.

7. **Visualization of Results**  
   - Visualize a subset of test images with predicted classes, marking tumor locations with red bounding boxes on detected cases.

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/brain_MRI_brain_tumor_detection_cnn.git
    cd Brain-MRI-Tumor-Detection
    ```

2. Run the Jupyter Notebook to execute each step.

## Results

The model reaches an accuracy of ~96% on the test set, with strong precision and recall scores for tumor detection. The final cell visualizes some example predictions with bounding boxes for detected tumors.

## Contributing

Feel free to submit issues or pull requests for improvements!
