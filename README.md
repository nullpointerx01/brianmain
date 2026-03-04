# 🧠 Brain Tumor Detection System
## DRDO Project - Medical Image Analysis using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

---

## 🎯 Overview

This project implements an automated brain tumor detection system using Convolutional Neural Networks (CNN) to classify MRI brain scans. The system can detect and classify brain tumors into four categories:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor (Healthy)**

---

## ✨ Features

- 🔬 **Deep Learning Model**: CNN-based architecture for accurate tumor classification
- 📊 **Data Augmentation**: Enhanced training with image augmentation techniques
- 🖥️ **Web Interface**: User-friendly Flask web application for predictions
- 📈 **Visualization**: Training metrics and prediction visualization
- 💾 **Model Persistence**: Save and load trained models
- 📱 **REST API**: API endpoints for integration with other systems

---

## 📁 Project Structure

```
brain_tumor_detection/
│
├── data/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── notumor/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── pituitary/
│       └── notumor/
│
├── models/
│   └── brain_tumor_model.h5
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── app/
│   ├── __init__.py
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── css/
│           └── style.css
│
├── notebooks/
│   └── brain_tumor_analysis.ipynb
│
├── utils/
│   ├── __init__.py
│   └── helpers.py
│
├── tests/
│   └── test_model.py
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd brain_tumor_detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

This project uses MRI brain scan images. You can use the following datasets:
- [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Brain Tumor Classification Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

Place the dataset in the `data/` directory following the structure mentioned above.

---

## 🏗️ Model Architecture

The CNN architecture consists of:
- **Input Layer**: 224x224x3 RGB images
- **Convolutional Blocks**: Multiple Conv2D + BatchNorm + MaxPool layers
- **Dense Layers**: Fully connected layers with dropout
- **Output Layer**: Softmax activation for 4-class classification

```
Model Summary:
- Total Parameters: ~2.5M
- Trainable Parameters: ~2.5M
- Input Shape: (224, 224, 3)
- Output: 4 classes (glioma, meningioma, pituitary, notumor)
```

---

## 🚀 Usage

### Training the Model
```bash
python src/train.py
```

### Making Predictions
```bash
python src/predict.py --image path/to/mri_image.jpg
```

### Running the Web Application
```bash
python app/app.py
```
Access the web interface at `http://localhost:5000`

### Using the API
```python
import requests

url = "http://localhost:5000/predict"
files = {"file": open("mri_scan.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~95% |
| Validation Accuracy | ~93% |
| Test Accuracy | ~92% |
| F1-Score | ~0.91 |

---

## 🔮 Future Enhancements

- [ ] Implement tumor segmentation using U-Net
- [ ] Add attention mechanisms for better interpretability
- [ ] Deploy on cloud (AWS/Azure)
- [ ] Mobile application integration
- [ ] Real-time detection from medical imaging devices
- [ ] Integration with hospital management systems

---

## 👥 Contributors

- DRDO Project Team

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- DRDO for project support
- Kaggle for datasets
- TensorFlow/Keras community

---

**⚠️ Disclaimer**: This system is intended for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis.
