
# Animal Species Detection

## Overview
This project focuses on detecting and classifying different animal species from images using advanced machine learning and deep learning techniques. The solution is designed to handle a variety of animal species and provides a user-friendly interface for easy use. It includes image preprocessing, feature extraction, model training, and an optional Streamlit-based web interface for real-time detection.

## Features
- **Preprocessing**: Image resizing, normalization, and data augmentation.
- **Deep Learning Models**: Utilizes transfer learning techniques with pre-trained models like ResNet, VGG, or custom CNN architectures for species classification.
- **Real-time Detection**: Detect species from images using a trained model.
- **Streamlit Web App**: A simple web interface for uploading images and detecting animal species.

## Getting Started

### Prerequisites
- Python 3.x
- Libraries: TensorFlow, Keras, OpenCV, Streamlit, NumPy, Pandas, Matplotlib
- You can install all the required libraries by running:
    ```bash
    pip install -r requirements.txt
    ```

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/swishtisingh/Animal_Species_Detection.git
    cd Animal_Species_Detection
    ```
2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset
This project uses animal species image datasets, which should be structured as follows:
You can either use your dataset or download one like the [Animal Faces dataset](https://www.kaggle.com/datasets) and place it in the `data/` directory.

## Model Training
1. Preprocess the dataset and train the model using:
    ```bash
    python train.py --dataset data/train --model-output model.h5
    ```
   - You can customize model parameters, such as learning rate and batch size, by adjusting the script.
   
2. Evaluate the model on the test set:
    ```bash
    python evaluate.py --dataset data/test --model model.h5
    ```

## Inference
To run inference on an image, use the detection script:
```bash
python detect.py --image-path path_to_image --model model.h5
This will output the predicted species label and display the image with the label.

Web App

You can also use the Streamlit-based web app for interactive species detection:

Run the app:
bash
Copy code
streamlit run app.py
Upload an image using the web interface and get real-time species detection results.
Project Structure

bash
Copy code
├── app.py              # Streamlit web app
├── detect.py           # Script for running inference on images
├── train.py            # Model training script
├── evaluate.py         # Model evaluation script
├── data/               # Directory for datasets
├── models/             # Directory for saving trained models
├── requirements.txt    # Required Python libraries
└── README.md           # Project documentation
Future Improvements

Add support for more animal species.
Integrate object detection for detecting multiple animals in a single image.
Improve model accuracy with hyperparameter tuning and data augmentation.
Contributing

Contributions are welcome! Feel free to submit pull requests for improvements or open issues if you encounter any bugs.

License

This project is licensed under the MIT License. See the LICENSE file for more details.
