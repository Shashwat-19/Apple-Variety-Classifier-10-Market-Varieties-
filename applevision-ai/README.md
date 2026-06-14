<p align="center">
  <img src="https://img.icons8.com/3d-fluency/94/apple.png" alt="AppleVision AI Logo" width="80"/>
</p>

<h1 align="center">🍎 AppleVision AI</h1>

<p align="center">
  <strong>Production-Grade AI-Powered Apple Variety Classification System</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-documentation">API Docs</a> •
  <a href="#deployment">Deployment</a> •
  <a href="#testing">Testing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12"/>
  <img src="https://img.shields.io/badge/Flask-3.x-black?logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="License"/>
  <img src="https://img.shields.io/badge/CI/CD-GitHub_Actions-2088FF?logo=github-actions&logoColor=white" alt="CI/CD"/>
  <img src="https://img.shields.io/badge/Coverage-80%25+-brightgreen" alt="Coverage"/>
</p>

---

## 📖 Overview

**AppleVision AI** is a commercial-grade web application that uses deep learning to classify apple varieties from uploaded images. Built with **EfficientNetV2S** transfer learning on the Fruits-360 dataset, it achieves **96%+ accuracy** across 10 market apple varieties with sub-second inference times.

The application features a stunning dark-mode SaaS UI, comprehensive REST API, prediction analytics dashboard, and full DevOps pipeline — designed to be portfolio-worthy, resume-ready, and production-deployable.

---

## ✨ Features

### 🤖 Machine Learning
- **EfficientNetV2S** architecture with transfer learning
- **10 apple variety** classification (market-relevant classes)
- **96%+ test accuracy** with sub-second inference
- Confidence scoring with high/medium/low indicators
- Top-K prediction probabilities with animated visualizations

### 🎨 Frontend
- **Premium SaaS-quality UI** inspired by Apple, OpenAI, and Google AI
- **Dark/Light mode** with seamless toggle (persisted in localStorage)
- **6 professional pages**: Home, Classify, About, Analytics, API Docs, Contact
- Drag-and-drop image upload with preview
- Animated confidence bars and real-time results
- Fully responsive mobile-first design
- Glassmorphism effects and micro-animations
- Chart.js analytics dashboards

### 🔧 Backend
- **Flask app factory** with clean architecture
- **RESTful API** with JSON responses and proper status codes
- **SQLAlchemy ORM** with prediction history storage
- **Rate limiting** (Flask-Limiter) and **CORS** support
- **Pydantic** request/response validation
- **Singleton ML service** with thread-safe model loading
- Comprehensive logging and error handling

### 🔐 Security
- Secure HTTP headers (nosniff, SAMEORIGIN, XSS protection)
- Rate limiting on prediction endpoints (100/hour)
- Input validation and file type verification
- CSRF and XSS protection
- No hardcoded secrets — everything via environment variables

### 📊 Analytics
- Total predictions counter
- Most predicted apple variety
- Daily/Weekly/Monthly prediction trends
- Average confidence metrics
- Interactive Chart.js visualizations

### 🐳 DevOps
- **Dockerfile** with non-root user and health checks
- **Docker Compose** with PostgreSQL service
- **GitHub Actions** CI/CD pipeline (lint → test → build)
- Automated test suite with 80%+ coverage target

---

## 🏗️ Architecture

```
applevision-ai/
│
├── app/                          # Flask application package
│   ├── __init__.py               # App factory (create_app)
│   ├── extensions.py             # SQLAlchemy, Limiter, CORS
│   ├── config/
│   │   └── settings.py           # Dev/Prod/Test configurations
│   ├── models/
│   │   └── prediction.py         # PredictionHistory ORM model
│   ├── schemas/
│   │   └── prediction.py         # Pydantic validation schemas
│   ├── routes/
│   │   ├── main_routes.py        # Page routes (/, /about, etc.)
│   │   └── api_routes.py         # REST API endpoints
│   ├── services/
│   │   ├── ml_service.py         # Singleton model loader & inference
│   │   ├── prediction_service.py # Database operations
│   │   └── analytics_service.py  # Statistics computation
│   ├── middleware/
│   │   └── security.py           # Security headers & request logging
│   ├── templates/                # Jinja2 templates (7 pages)
│   └── static/                   # CSS, JavaScript assets
│
├── tests/                        # Pytest test suite
│   ├── conftest.py               # Fixtures & mocks
│   ├── test_api.py               # API endpoint tests
│   ├── test_services.py          # Service layer tests
│   └── test_models.py            # Database model tests
│
├── .github/workflows/ci.yml     # GitHub Actions pipeline
├── Dockerfile                    # Production container
├── docker-compose.yml            # Multi-service orchestration
├── requirements.txt              # Python dependencies
├── run.py                        # Development entry point
├── wsgi.py                       # Gunicorn WSGI entry
└── .env.example                  # Environment variable template
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.12, Flask 3.x, Gunicorn |
| **ML/AI** | TensorFlow 2.15+, EfficientNetV2S, NumPy, Pillow |
| **Database** | SQLAlchemy ORM, SQLite (dev), PostgreSQL (prod) |
| **Frontend** | HTML5, Tailwind CSS, Vanilla JavaScript, Chart.js |
| **Validation** | Pydantic v2, Marshmallow |
| **Security** | Flask-CORS, Flask-Limiter, Secure Headers |
| **DevOps** | Docker, Docker Compose, GitHub Actions |
| **Testing** | pytest, pytest-cov, pytest-flask |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- pip
- (Optional) Docker & Docker Compose

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties-.git
cd Apple-Variety-Classifier-10-Market-Varieties-/applevision-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings (defaults work for local dev)

# 5. Run the application
python run.py
```

The app will be available at **http://localhost:5000**

### Docker

```bash
# Build and run with Docker
docker build -t applevision-ai .
docker run -p 8000:8000 applevision-ai

# Or use Docker Compose (includes PostgreSQL)
docker-compose up
```

The app will be available at **http://localhost:8000**

---

## 📡 API Documentation

### Base URL
```
http://localhost:8000/api
```

### Endpoints

#### 🔍 Health Check
```http
GET /api/health
```
Returns application health status, model state, and uptime.

#### 🍎 Classify Apple
```http
POST /api/predict
Content-Type: multipart/form-data
```
| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | File | Apple image (JPG, PNG, WEBP) |

**Response:**
```json
{
  "top_class": "Apple 10",
  "confidence": 0.9732,
  "inference_time_ms": 42.5,
  "is_high_confidence": true,
  "top_predictions": [
    {"class_name": "Apple 10", "score": 0.9732},
    {"class_name": "Apple 13", "score": 0.0121}
  ],
  "threshold": 0.75
}
```

#### 📜 Prediction History
```http
GET /api/history?page=1&per_page=20&search=Apple&sort_by=timestamp&sort_order=desc
```

#### 📊 Analytics
```http
GET /api/stats
```
Returns prediction statistics: totals, distribution, trends, and averages.

---

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-flask

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_api.py -v

# Run with HTML coverage report
pytest tests/ --cov=app --cov-report=html
```

**Coverage Target:** 80%+

---

## 🚢 Deployment

### Render

1. Connect your GitHub repository to Render
2. Create a new **Web Service**
3. Set the following:
   - **Root Directory:** `applevision-ai`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn wsgi:app -w 2 -b 0.0.0.0:$PORT`
4. Add environment variables from `.env.example`

### Railway

1. Connect your GitHub repository
2. Set the **Root Directory** to `applevision-ai`
3. Railway auto-detects Python and Gunicorn
4. Add environment variables in the Railway dashboard

### AWS EC2

```bash
# SSH into your instance
ssh -i your-key.pem ec2-user@your-ip

# Install Docker
sudo yum install -y docker
sudo systemctl start docker

# Clone and deploy
git clone https://github.com/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties-.git
cd Apple-Variety-Classifier-10-Market-Varieties-/applevision-ai
docker-compose up -d
```

### DigitalOcean

1. Create a Droplet (Ubuntu 22.04, 2GB RAM minimum)
2. SSH in and install Docker + Docker Compose
3. Clone the repo and run `docker-compose up -d`
4. Configure your domain with a reverse proxy (Nginx/Caddy)

### Docker VPS (Generic)

```bash
# On any VPS with Docker installed
git clone https://github.com/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties-.git
cd Apple-Variety-Classifier-10-Market-Varieties-/applevision-ai
cp .env.example .env
# Edit .env with production values
docker-compose up -d
```

---

## 🍎 Supported Apple Varieties

| # | Variety |
|---|---------|
| 1 | Apple 10 |
| 2 | Apple 13 |
| 3 | Apple 18 |
| 4 | Apple 19 |
| 5 | Apple 7 |
| 6 | Apple 8 |
| 7 | Apple 9 |
| 8 | Apple Red Yellow 2 |
| 9 | Apple hit 1 |
| 10 | Apple worm 1 |

---

## 🔮 Future Enhancements

- [ ] 🔐 User authentication (JWT / OAuth2)
- [ ] 📸 GradCAM explainability visualizations
- [ ] 🌐 Multi-language support (i18n)
- [ ] 📱 Progressive Web App (PWA) support
- [ ] 🔄 Model versioning and A/B testing
- [ ] 📈 Real-time WebSocket predictions
- [ ] 🗄️ Redis caching layer
- [ ] 📊 Prometheus + Grafana monitoring
- [ ] 🤖 Batch prediction API
- [ ] ☁️ TensorFlow Serving integration

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code:
- Passes all existing tests
- Includes tests for new features
- Follows PEP 8 style guidelines
- Has proper docstrings and type hints

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](../LICENSE) file for details.

---

## 📬 Contact

**Shashwat** — [@Shashwat-19](https://github.com/Shashwat-19)

Project Link: [https://github.com/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties-](https://github.com/Shashwat-19/Apple-Variety-Classifier-10-Market-Varieties-)

---

<p align="center">
  Built with ❤️ using TensorFlow, Flask, and modern web technologies.
  <br/>
  <strong>AppleVision AI</strong> — Where AI meets Agriculture 🍎
</p>
