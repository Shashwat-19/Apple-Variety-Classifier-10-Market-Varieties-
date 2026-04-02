# 🍎 Apple Variety Classifier

> **An ultra-premium, AI-powered web application for classifying 10 market varieties of apples with confidence-aware inference.**

[![GitHub stars](https://img.shields.io/github/stars/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties?style=social)](https://github.com/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties)](https://github.com/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties/issues)
[![License](https://img.shields.io/github/license/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties)](./LICENSE)

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](#)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?logo=flask&logoColor=black)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Model](https://img.shields.io/badge/Model-EfficientNetV2S-blueviolet)](#)
[![Validation Accuracy](https://img.shields.io/badge/Val%20Accuracy-96.22%25-brightgreen)](#)

---

## ✨ Overview

Accurate identification of apple varieties in real-world market conditions is challenging due to high visual similarity across cultivars, varying lighting conditions, and subtle texture differences. Manual inspection is time-consuming and prone to error.

This project delivers a **state-of-the-art deep learning system** wrapped in an **ultra-premium, Apple-inspired web dashboard**. By combining an EfficientNetV2S neural network with a frosted-glass frontend, users can effortlessly drag-and-drop apple specimens to receive instant, confidence-aware classifications.

---

## 🚀 Key Features

### 🖥️ Premium Web Interface
- **Apple-Inspired Aesthetic:** A sleek, modern design featuring glassmorphism (acrylic blur effects), refined typography (Inter), and subtle radial background gradients.
- **Dynamic Visualizations:** Custom, stagger-animated CSS progress bars that elegantly fill up to represent class probabilities.
- **Fluid Interactions:** An interactive drag-and-drop zone with responsive hover states and real-time inference feedback.
- **SaaS-Grade Toasts:** A modern error notification system engineered for a flawless user experience.

### 🧠 Robust AI Engine
- **EfficientNetV2S Backbone:** Utilizes a highly optimized, ImageNet-pretrained model for unparalleled accuracy-efficiency trade-offs.
- **Confidence-Aware Filtering:** Predictions are only delivered if the system's confidence exceeds **75%** (98.1% inference reliability), eliminating forced false positives on ambiguous images.
- **Two-Stage Transfer Learning:** Trained utilizing frozen layers transitioning into fine-tuning for top-tier generalization across 10 visually similar apple cultivars.

---

## ⚙️ Model Architecture & Training

The core AI engine is built with TensorFlow/Keras and features:

1. **Feature Extraction:** A frozen EfficientNetV2S backbone to capture mid-to-high-level visual features like edges, complex textures, and color variances.
2. **Custom Classification Head:** Global Average Pooling combined with Dense layers and a Softmax output layer tailored to our 10 specific apple classes.
3. **Training Strategy:** 
   - *Stage 1:* 60 Epochs (Head only, Adam optimizer at `3e-4`)
   - *Stage 2:* 80 Epochs (Top 25% of backbone unfrozen, learning rate reduced to `1e-5` for granular fine-tuning).

**Final Performance Setup:**
- Validation Accuracy: **96.22%**
- High-Confidence Predictions: **98.1%**
- Test Inference Time: **Ultra-fast**

---

## 🛠️ Installation & Local Development

Run the web application locally on your machine with a few simple steps.

### 1. Clone the repository
```bash
git clone https://github.com/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties.git
cd Apple-Variety-Classifier-10-Market-Varieties
```

### 2. Create a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt flask pillow
```

### 4. Launch the Web Dashboard
```bash
python3 app.py
```
> **Success:** Open your web browser and navigate to `http://127.0.0.1:5001` to use the application!

---

## 🔮 Future Roadmap

- [ ] Mobile deployment via **TensorFlow Lite**.
- [ ] Integration of **Grad-CAM visual explanations** to show users *why* the AI made its decision.
- [ ] Expansion of the dataset to include rare regional apple cultivars.
- [ ] Containerization via **Docker** for one-click deployments.

---

## 📄 License & Documentation

This project is open-source and distributed under the [MIT License](./LICENSE). 

For expansive technical documentation on the AI model and its origins, check out my articles on [Hashnode](https://hashnode.com/@Shashwat56).

---

## 👨‍💻 About the Author

### **Shashwat**
**Java Developer | Cloud & NoSQL Enthusiast | UI/UX Innovator**

I specialize in building robust backend systems and wrapping them in highly engaging, scalable, and visually impressive user interfaces. 

🔹 **Languages:** Java, Python, JavaScript  
🔹 **Technologies:** Spring Boot, Flask, TensorFlow  
🔹 **Cloud & Systems:** AWS, Docker, MongoDB, Firebase  

### Let's Connect!
[<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />](https://github.com/Shashwat-19) 
[<img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/shashwatk1956/) 
[<img src="https://img.shields.io/badge/Hashnode-2962FF?style=for-the-badge&logo=hashnode&logoColor=white" />](https://hashnode.com/@Shashwat56)
[<img src="https://img.shields.io/badge/HackerRank-15%2B-2EC866?style=for-the-badge&logo=HackerRank&logoColor=white" />](https://www.hackerrank.com/profile/shashwat1956)

*Feel free to reach out for tech collaborations, open-source engineering, or to chat about building innovative AI tools!*
