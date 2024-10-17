
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
```bash
pip install -r requirements.txt
```

5. Run the Application
Once all dependencies are installed, launch the Streamlit web app for real-time animal detection:
```bash
streamlit run './scripts/app.py'
```

## Model Training

The model training is performed using YOLOv8 for object detection, which is known for its real-time performance and high accuracy. The configuration files for the model can be found in the config directory, and you can run the training by using the provided Jupyter notebook yolov8.ipynb.

### Steps for Training:
1. Prepare the dataset by placing images and their corresponding labels in the data/ directory.
2. Use the train_test_split.py script to split the dataset into training and testing sets.
3. Modify the custom.yaml file to reflect the path to the dataset.
4. Open and run yolov8.ipynb to train the model.

## Evaluation
After training the model, it is evaluated on standard object detection metrics, including Precision, Recall, and Mean Average Precision (mAP). The model performs well on both individual species detection and overall mAP scores.

| Model   | Precision | Recall | F1-score | mAP@0.5 | mAP@0.5:0.95 |
|---------|-----------|--------|----------|---------|--------------|
| YOLOv8  |   0.944   |  0.915 |   0.93   |   0.95  |    0.804     |

These results showcase the efficiency of the model in detecting wildlife in real-time scenarios.

## Contributing

Contributions to this project are highly encouraged! Whether you're fixing bugs, improving documentation, or adding new features, every contribution helps. If you'd like to contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes.
4. Commit and push your changes (git push origin feature-branch).
5. Open a Pull Request on GitHub.
Please make sure your code adheres to the PEP 8 style guidelines.

## Author

### Srishti Singh
Data Scientist and Machine Learning Enthusiast
Feel free to connect with me on [LinkedIn.](https://www.linkedin.com/in/srishti-singh-921aa52aa/)

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## Happy Coding!

