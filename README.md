<div align="center">

# 🍎 Apple Variety Classifier

### Confidence-Aware Deep Learning for Real-World Apple Cultivar Recognition

**EfficientNetV2S · Two-Stage Transfer Learning · Flask · Serverless Deployment on Vercel**

[![Live Demo](https://img.shields.io/badge/demo-live-success?style=for-the-badge&logo=vercel&logoColor=white)](https://apple-variety-classifier-10-market.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](#)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=flat-square&logo=flask&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker&logoColor=white)](#)
[![Deployed on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=flat-square&logo=vercel&logoColor=white)](https://apple-variety-classifier-10-market.vercel.app)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](LICENSE)

[**Live Demo**](https://apple-variety-classifier-10-market.vercel.app) · [**Report Bug**](https://github.com/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties-/issues) · [**Request Feature**](https://github.com/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties-/issues)

</div>

---

## Overview

Apple cultivars are visually near-identical across variety boundaries — overlapping color, texture, and shape distributions make naive classifiers unreliable at inference time, especially under uncontrolled lighting and background conditions typical of real markets.

This project implements a **10-class apple variety classifier** built on an **EfficientNetV2S backbone with two-stage transfer learning**, wrapped in a **confidence-gated inference layer** that suppresses low-confidence predictions instead of forcing a guess. The model is served through a Flask backend and deployed as a serverless function on Vercel, with a separate Streamlit build used for rapid experimentation.

> **Design principle:** a wrong "I don't know" is cheaper than a wrong confident answer. The system only returns a class label when calibrated confidence clears a 75% threshold — everything else is flagged as ambiguous rather than silently misclassified.

---

## Key Results

| Metric | Value |
|---|---|
| Validation Accuracy | **96.22%** |
| High-Confidence Prediction Rate | **98.1%** (predictions above the 75% confidence gate) |
| Classes | 10 market apple varieties |
| Backbone | EfficientNetV2S (ImageNet-pretrained) |
| Training Strategy | 2-stage transfer learning (60 + 80 epochs) |

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐     ┌────────────────┐
│  Image Input │ --> │ EfficientNetV2S   │ --> │ Classification Head│ --> │ Confidence Gate │
│ (drag/drop)  │     │ (frozen backbone) │     │ GAP + Dense+Softmax│     │  ≥ 75% → label  │
└─────────────┘     └──────────────────┘     │                    │     │  < 75% → reject │
                                              └───────────────────┘     └────────────────┘
```

**Training pipeline**

1. **Stage 1 — Feature extraction:** EfficientNetV2S backbone frozen; only the classification head (Global Average Pooling → Dense → Softmax) is trained for 60 epochs, Adam @ `3e-4`.
2. **Stage 2 — Fine-tuning:** top 25% of the backbone unfrozen, learning rate dropped to `1e-5` for 80 epochs to adapt high-level features to the 10 target cultivars without catastrophic forgetting.
3. **Inference:** softmax output is passed through a confidence gate — predictions below 75% probability are marked as ambiguous rather than returned as a hard label, which is what drives the 98.1% high-confidence reliability figure above.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | TensorFlow / Keras, EfficientNetV2S |
| Backend | Flask (production WSGI app) |
| Prototyping | Streamlit (separate experimentation build) |
| Frontend | HTML/CSS/JS, glassmorphism UI, drag-and-drop upload |
| Deployment | Vercel (serverless Python function via `@vercel/python`) |
| Containerization | Docker |
| Training Environment | Jupyter Notebook |

---

## Project Structure

```
Apple-Variety-Classifier-10-Market-Varieties-/
├── applevision-ai/                          # Production Flask app (deployed via Vercel)
│   ├── wsgi.py                              # WSGI entrypoint used by vercel.json
│   └── app/static/                          # Static assets served in production
├── apple-variety-streamlit/                 # Streamlit build for rapid iteration/demos
├── static/ , templates/                     # Flask app UI assets
├── Apple_Variety_Classifier_(10_Market_Varieties).ipynb   # Training pipeline (EfficientNetV2S, 2-stage transfer learning)
├── app.py                                   # Local Flask entrypoint
├── Dockerfile                               # Container build for local/self-hosted deployment
├── vercel.json                              # Serverless routing config for Vercel
├── requirements.txt                         # Base runtime dependencies
├── requirements.flask.txt                   # Flask-app-specific dependencies
└── requirements.training.txt                # Training/notebook dependencies
```

---

## Getting Started

### Prerequisites
- Python 3.x
- pip / venv

### Local Setup

```bash
# 1. Clone
git clone https://github.com/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties-.git
cd Apple-Variety-Classifier-10-Market-Varieties-

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt -r requirements.flask.txt

# 4. Run the app
python3 app.py
```

Open `http://127.0.0.1:5001` in your browser.

### Run with Docker

```bash
docker build -t apple-classifier .
docker run -p 5001:5001 apple-classifier
```

### Production Deployment (Vercel)

The app is deployed as a serverless Python function. `vercel.json` routes all traffic through `applevision-ai/wsgi.py`, with static assets served separately from `applevision-ai/app/static`. Pushing to `main` on a Vercel-linked repo redeploys automatically — no manual build step required.

---

## Roadmap

- [ ] Mobile deployment via TensorFlow Lite
- [ ] Grad-CAM visual explanations for prediction interpretability
- [ ] Expand dataset to rare/regional apple cultivars
- [ ] CI pipeline for automated model regression testing on push

---

## 📚 Documentation

Comprehensive documentation for this project is available on [Hashnode](https://hashnode.com/@Shashwat56).

> At present, this README serves as the primary source of documentation.

## 📜 License

This project is distributed under the MIT License.  
For detailed licensing information, please refer to the [LICENSE](./LICENSE) file included in this repository.

## 📩 Contact  
## Shashwat

**Machine Learning Engineer | Scalable AI Systems**

🔹 **ML systems:** (CV, NLP) + data pipelines<br>
🔹 **End-to-end:** training → deployment<br>
🔹 **Backend & Cloud:** Python, Flask, Node.js, Docker, AWS<br>
🔹 **Projects:** Traffic AI, Video Summarizer, AI Assistants<br>

---

## 🚀 Open Source | Tech Innovation  
Building robust applications and leveraging cloud technologies for high-performance solutions.

---

### 📌 Find me here:  
[<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />](https://github.com/Shashwat-19)  [<img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/shashwatk1956/)  [<img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" />](mailto:shashwat1956@gmail.com)  [<img src="https://img.shields.io/badge/Hashnode-2962FF?style=for-the-badge&logo=hashnode&logoColor=white" />](https://hashnode.com/@Shashwat56)
[<img src="https://img.shields.io/badge/HackerRank-15%2B-2EC866?style=for-the-badge&logo=HackerRank&logoColor=white" />](https://www.hackerrank.com/profile/shashwat1956)

> Feel free to connect for tech collaborations, open-source contributions, or brainstorming innovative solutions!
>