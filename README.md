# ASL Alphabet Detection using Deep Learning

## Project Overview
This project focuses on real-time detection of American Sign Language (ASL) alphabets using Deep Learning and Computer Vision.

A Convolutional Neural Network based on MobileNetV2 is trained on ASL image data to classify hand gestures into alphabet classes. The trained model is then deployed using OpenCV to perform real-time predictions via webcam.

---

## Objectives
- Build an image classification model for ASL alphabets
- Apply transfer learning using MobileNetV2
- Perform real-time detection using webcam
- Understand deployment of deep learning models

---

## Model Architecture
- Base Model: MobileNetV2 (pretrained on ImageNet)
- Global Average Pooling
- Dense Layer (256 units)
- Dropout Layer
- Output Layer (classification)

---

## Dataset
- ASL Alphabet Dataset (Kaggle)
- Contains images for 26 alphabets (A–Z)
- Images resized and normalized before training

---

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Google Colab

---

## How to Run (VS Code)

### 1. Clone Repository
```bash
git clone https://github.com/your-username/ASL-Detection-DeepLearning.git
cd ASL-Detection-DeepLearning
```
### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run Application
```bash
python main.py
```
---
## Real-Time Detection
Webcam captures live video

Region of Interest (ROI) is extracted

Image is preprocessed and passed to model

Predicted alphabet is displayed on screen

---
## Limitations
Model may show high confidence due to limited training steps

No "No Sign" class included in dataset

Performance depends on lighting and background

---
## Future Improvements
Add "No Hand / No Sign" class

Improve dataset diversity

Increase training epochs

Implement word/sentence formation

Add voice output

---
## Conclusion

This project demonstrates the practical implementation of deep learning for real-time gesture recognition. It highlights the complete pipeline from data preprocessing and model training to deployment using computer vision techniques.

---
## Project Resources

- **Google Colab Notebook:** [Open Notebook](https://colab.research.google.com/drive/1kSdUbbHbcmmP8fvaLUzoTqg3ikrD_6sp?usp=sharing)
- **ASL Alphabet Dataset:** [Open Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data.)


## Author
Irtika.
