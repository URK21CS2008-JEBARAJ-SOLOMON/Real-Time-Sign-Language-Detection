---

# Real-Time Sign Language Detection Using CNN

## 📜 Project Overview
This project presents a real-time sign language detection system that bridges the communication gap for the hearing and speech-impaired community. The system leverages a Convolutional Neural Network (CNN) classifier trained on a custom dataset of hand gestures representing alphabet signs (A-Z). Efficient hand tracking, preprocessing, and gesture recognition modules work together to deliver fast, accurate predictions through a webcam interface.

---

## 🛠️ Features
- Real-time detection of hand gestures via live webcam feed
- Dynamic hand tracking and localization using the cvzone HandDetector module
- Preprocessing pipeline for cropping, resizing, normalization, and background uniformity
- Gesture classification using a Teachable Machine-trained CNN model
- Minimal latency with average frame processing time around 50ms
- Robust performance across variations in lighting, orientation, and backgrounds

---

## ⚙️ Technologies Used
- Python 3
- OpenCV
- cvzone Library
- TensorFlow/Keras
- Teachable Machine (Google)
- NumPy
- scikit-learn (for evaluation metrics)

---

## 📂 Project Structure
```
├── datacollection.py       # Script for capturing and saving hand gesture images
├── test.py                 # Real-time gesture detection and prediction
├── keras_model.h5          # Trained CNN model
├── labels.txt              # Labels corresponding to gestures (A-Z)
├── Data/                   # Directory storing captured images (organized by class)
├── README.md               # Project documentation
```

---

## 🚀 How to Run
1. Clone the repository or download the project files.
2. Ensure all dependencies are installed:
   ```
   pip install opencv-python cvzone tensorflow scikit-learn numpy
   ```
3. Run the real-time detection script:
   ```
   python test.py
   ```
4. Show hand gestures in front of your webcam to see live predictions!

---

## 📊 Performance
- **Accuracy:** ~93% on test set
- **Precision:** ~90%
- **Recall:** ~88%
- **F1 Score:** ~89%

> Metrics are computed based on manually captured test data across various environmental conditions.

---

## 🔥 Future Improvements
- Extend gesture vocabulary to include dynamic gestures and full words
- Optimize the system for deployment on mobile and embedded devices
- Incorporate adaptive illumination handling for extreme lighting scenarios

---

## 🙏 Acknowledgements
- [Teachable Machine by Google](https://teachablemachine.withgoogle.com/)
- [cvzone Library](https://github.com/cvzone/cvzone)
- OpenCV and TensorFlow communities for extensive support and documentation

---
