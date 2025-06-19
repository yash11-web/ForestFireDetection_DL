# Forest Fire Detection Using Deep Learning

This project aims to detect forest fires from images using a Convolutional Neural Network (CNN). The model is trained on a labeled dataset of fire and non-fire images and deployed through a user-friendly Gradio interface for real-time predictions.

## Project Overview

Forest fires are a major threat to the environment, wildlife, and human life. Early detection is crucial for minimizing damage. This project uses deep learning to automatically classify images as either **"Fire"** or **"No Fire"**, helping enable faster response and disaster mitigation.

##  Key Features

- Built a CNN using TensorFlow and Keras for image classification.
- Used ImageDataGenerator for preprocessing, augmentation, and better generalization.
- Evaluated the model using precision, recall, F1-score, and confusion matrix.
- Deployed a Gradio-based web interface for real-time image predictions.

## Tech Stack

- **Language:** Python  
- **Deep Learning:** TensorFlow, Keras  
- **Data Handling:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn  
- **Evaluation:** Scikit-learn  
- **Deployment:** Gradio  
- **Development:** Google Colab  
- **Dataset:** [Kaggle - The Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset)


## Model Architecture

- Input Layer (150x150x3)
- Conv2D → MaxPooling
- Conv2D → MaxPooling
- Conv2D → MaxPooling
- Flatten
- Dense (ReLU) → Dropout
- Output Layer (Sigmoid)

## Evaluation Metrics

- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **Validation Curve (Accuracy vs Epochs)**

## Gradio Interface

A simple web UI where users can upload an image and get a prediction (`Fire` or `No Fire`) along with confidence percentage.

