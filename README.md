# Apple Variety Classifier

### EfficientNetV2S with Confidence-Aware Inference (10 Market Varieties)

[![GitHub stars](https://img.shields.io/github/stars/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties?style=social)](https://github.com/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties?style=social)](https://github.com/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties/network)
[![GitHub issues](https://img.shields.io/github/issues/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties)](https://github.com/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties)](https://github.com/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties/pulls)
[![License](https://img.shields.io/github/license/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties)](./LICENSE)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Model](https://img.shields.io/badge/Model-EfficientNetV2S-blueviolet)](#)
[![Classes](https://img.shields.io/badge/Classes-10-blue)](#)
[![Validation Accuracy](https://img.shields.io/badge/Val%20Accuracy-96.22%25-brightgreen)](#)
[![Confidence Aware](https://img.shields.io/badge/Inference-Confidence%20Aware-success)](#)

[![Platform](https://img.shields.io/badge/platform-Windows%20|%20macOS%20|%20Linux-blue)](#)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Open Source](https://img.shields.io/badge/Open%20Source-Yes-green)](#)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen)](#)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties/blob/main/notebooks/01_data_exploration.ipynb)

---

## Overview

Accurate identification of apple varieties in real-world market conditions is challenging due to high visual similarity across cultivars, varying lighting conditions, and subtle texture differences. Manual inspection is time-consuming and prone to error.

This project presents a **deep learning–based apple variety classification system** built using **EfficientNetV2S**, trained with a **two-stage transfer learning strategy** and enhanced with **confidence-aware inference** to ensure reliable predictions in deployment scenarios.

The model classifies **10 commonly sold apple varieties** and delivers predictions only when the confidence score exceeds a defined threshold, reducing false positives and improving real-world usability.

---

## Key Features

- Classification of **10 apple market varieties**
- **EfficientNetV2S** backbone (ImageNet pretrained)
- Two-stage training (frozen + fine-tuning)
- Confidence-aware prediction filtering
- High inference reliability (98.1% high-confidence predictions)
- Robust performance across all classes
- Deployment-ready architecture

---

## Model Architecture

The model employs a **transfer learning–based architecture** using **EfficientNetV2S**, pretrained on ImageNet, chosen for its strong accuracy–efficiency trade-off and stable training behavior.

### Backbone

- **EfficientNetV2S (ImageNet pretrained)**
- Acts as a high-capacity feature extractor
- Captures robust low- and mid-level visual features (edges, textures, color patterns)

### Classification Head

- **Global Average Pooling** to reduce spatial dimensions and overfitting
- **Fully Connected Layers** for class-specific feature learning
- **Softmax Output Layer** producing probabilities for **10 apple varieties**

### Training Strategy

A **two-stage training approach** is used:

- **Stage 1:** Backbone frozen; only the classification head is trained to learn dataset-specific class boundaries
- **Stage 2:** Top **25% of backbone layers** are unfrozen and fine-tuned with a lower learning rate to adapt high-level features

This strategy ensures **stable convergence**, **reduced overfitting**, and **strong generalization** across all classes.

## Experimental Setup

| Parameter                  | Value                                               |
| -------------------------- | --------------------------------------------------- |
| Architecture               | EfficientNetV2S + Custom Head                       |
| Number of Classes          | 10                                                  |
| Stage-1 Epochs (Frozen)    | 60                                                  |
| Stage-2 Epochs (Fine-tune) | 80                                                  |
| Batch Size                 | 32                                                  |
| Image Size                 | 224 × 224                                           |
| Optimizer                  | Adam                                                |
| Stage-1 Learning Rate      | 3e-4                                                |
| Stage-2 Learning Rate      | 1e-5                                                |
| Loss Function              | Sparse Categorical Crossentropy                     |
| Augmentation               | Flip, Rotate, Zoom, Brightness, Contrast, Translate |
| Backbone Trainable         | Top 25% layers (Stage-2)                            |
| Confidence Threshold       | 0.75                                                |
| Validation Split           | Balanced dataset                                    |

---

## Training & Validation Analysis

The training process demonstrates stable convergence with strong generalization:

- Training and validation accuracy remain closely aligned
- A brief accuracy dip occurs during the transition from Stage-1 to Stage-2 fine-tuning (around epoch ~60)
- This dip corresponds to newly unfrozen layers adapting to task-specific features
- Performance stabilizes and improves after adaptation

### Final Performance Summary

```

================================================================================
FINAL MODEL SUMMARY — Apple EfficientNetV2S
===========================================

Total Epochs Trained                38

Final Training Accuracy             0.9883
Final Validation Accuracy           0.9622
Best Validation Accuracy            0.9744
------------------------------------------

Final Training Loss                 0.036
Final Validation Loss               0.1341
------------------------------------------

Confidence Threshold                0.75
High-Confidence Predictions         151 / 154 (98.1%)
Low-Confidence Rejections           3 / 154 (1.9%)
Mean Confidence Score               0.989
-----------------------------------------

Inference Reliability
• False positives reduced using confidence-aware filtering
• Predictions delivered only when confidence ≥ 0.75
• Stable performance across all 10 apple varieties
==================================================

```

---

## Confidence-Aware Inference

To improve real-world reliability, the model applies **confidence-aware filtering** during inference:

- Predictions are accepted only if confidence ≥ **0.75**
- Low-confidence predictions are rejected rather than forced
- Reduces false positives in visually ambiguous cases
- Ensures only reliable predictions are delivered to end-users

This approach significantly improves deployment safety compared to standard softmax-based classification.

---

## Dataset

- Balanced dataset of apple images
- 10 visually similar apple varieties
- Images resized to **224 × 224**
- Data augmentation applied to improve robustness
- Dataset not included in the repository due to size constraints

---

## Installation & Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python src/train.py
```

### Evaluate with Confidence Filtering

```bash
python src/evaluate.py
```

---

## Results & Visualizations

The following evaluation artifacts are generated:

- Training vs Validation Accuracy Curves
- Training vs Validation Loss Curves
- Confusion Matrix
- Confidence Score Distribution
- High-confidence vs rejected predictions

All outputs are stored in the `results/` directory.

---

## Future Improvements

- Grad-CAM visual explanations
- Mobile deployment using TensorFlow Lite
- Domain adaptation for different lighting environments
- Real-time webcam inference
- API deployment using FastAPI

---

## License

This project is licensed under the **MIT License**.

---

## Acknowledgements

- TensorFlow & Keras team for EfficientNetV2
- Open-source community for datasets and tools

---

## Documentation

Comprehensive documentation for this project is available on [Hashnode](https://hashnode.com/@Shashwat56).

> At present, this README serves as the primary source of documentation.

## License

This project is distributed under the MIT License.  
For detailed licensing information, please refer to the [LICENSE](./LICENSE) file included in this repository.

## Contact

### Shashwat

**Java Developer | Cloud & NoSQL Enthusiast**

🔹 **Java** – OOP, Backend Systems, APIs, Automation  
🔹 **Cloud & NoSQL** – Docker, AWS, MongoDB, Firebase Firestore  
🔹 **UI/UX Design** – Scalable, user-focused, and visually engaging apps

---

## Open Source | Tech Innovation

Building robust applications and leveraging cloud technologies for high-performance solutions.

---

### Find me here:

[<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />](https://github.com/Shashwat-19) [<img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/shashwatk1956/) [<img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" />](mailto:shashwat1956@gmail.com) [<img src="https://img.shields.io/badge/Hashnode-2962FF?style=for-the-badge&logo=hashnode&logoColor=white" />](https://hashnode.com/@Shashwat56)
[<img src="https://img.shields.io/badge/HackerRank-15%2B-2EC866?style=for-the-badge&logo=HackerRank&logoColor=white" />](https://www.hackerrank.com/profile/shashwat1956)

Feel free to connect for tech collaborations, open-source contributions, or brainstorming innovative solutions!

---
