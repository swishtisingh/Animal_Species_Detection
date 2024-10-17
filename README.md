
# Real-Time Animal Species Detection

The aim of this project is to develop an efficient computer vision model capable of real-time wildlife detection. Leveraging state-of-the-art deep learning techniques, the model can accurately identify animal species in live video streams or images, helping researchers, conservationists, and enthusiasts monitor wildlife.

<p align="center">
  <img src="./demo/demo.gif" alt="Demo GIF">
</p>

## Table of Contents
- [Overview](#overview)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Web App](#web-app)
- [Contirbuting](#contributing)
- [Author](#author)
- 
## Overview
The Real-Time Animal Species Detection project focuses on detecting 10 different animal species, including buffalo, cheetahs, elephants, and more, using advanced computer vision techniques. The main objective is to build a highly accurate model that runs in real-time, enabling its deployment in wildlife monitoring systems. The project uses the popular YOLO (You Only Look Once) model, which is known for its efficiency in object detection tasks. The codebase includes data preprocessing scripts, training scripts, and deployment utilities.

## Datasets
This project uses datasets containing labeled images of 10 animal species. Below are the datasets utilized, which consist of thousands of images for training and validation purposes:
- Dataset1: [African Wildlife Dataset](https://www.kaggle.com/datasets/biancaferreira/african-wildlife)
- Dataset2: [Danger of Extinction Animal Image Set](https://www.kaggle.com/datasets/brsdincer/danger-of-extinction-animal-image-set)
- Dataset3: [Animals Detection Images Dataset](https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset )
These datasets provide rich diversity in backgrounds, lighting conditions, and poses, which helps the model generalize well to various real-world scenarios.

## Features
- **Preprocessing**: Image resizing, normalization, and data augmentation.
- **Deep Learning Models**: Utilizes transfer learning techniques with pre-trained models like ResNet, VGG, or custom CNN architectures for species classification.
- **Real-time Detection**: Detect species from images using a trained model.
- **Streamlit Web App**: A simple web interface for uploading images and detecting animal species.

## Project Structure
The project follows a modular and organized structure for ease of use and collaboration. Below is the directory layout of the project:
```bash
    ├── config
    │   └── custom.yaml    
    ├── data
    │   ├── images         
    │   └── labels         
    ├── logs
    │   └── log.log      
    ├── notebooks
    │   └── yolov8.ipynb
    ├── runs
    │   └── detect
    │       ├── train
    │       └── val
    ├── scripts
    │   ├── app.py
    │   ├── convert_format.py
    │   └── train_test_split.py
    ├── README.md
    └── requirements.txt
```
Each directory is clearly categorized to help you quickly locate the necessary scripts and data.

## Getting Started
Follow these steps to set up the environment and run the application locally.

### 1. Fork and Clone the Repository
First, fork the repository from GitHub and clone it to your local machine:
```bash
git clone https://github.com/<YOUR-USERNAME>/Animal-Species-Detection
cd Animal-Species-Detection
```
### 2. Create a Virtual Environment
Use Python's venv module to create a virtual environment:
```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment
On Linux/macOS:
```bash
source venv/bin/activate
```
On Windows:
```bash
venv\Scripts\activate
```

### 4. Install Dependencies
Install the required Python libraries using the requirements.txt file:





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
```
##Web App

You can also use the Streamlit-based web app for interactive species detection:

1. Run the app:
```bash
streamlit run app.py
```
2. Upload an image using the web interface and get real-time species detection results.

## Project Structure

```bash

├── app.py              # Streamlit web app
├── detect.py           # Script for running inference on images
├── train.py            # Model training script
├── evaluate.py         # Model evaluation script
├── data/               # Directory for datasets
├── models/             # Directory for saving trained models
├── requirements.txt    # Required Python libraries
└── README.md           # Project documentation
```
## Future Improvements

- Add support for more animal species.
- Integrate object detection for detecting multiple animals in a single image.
- Improve model accuracy with hyperparameter tuning and data augmentation.

## Contributing

Contributions are welcome! Feel free to submit pull requests for improvements or open issues if you encounter any bugs.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
