# 🍎 Apple Variety Classifier

A production-ready Streamlit application for classifying apple varieties using Deep Learning (EfficientNetV2S). Designed with a modern, Apple-inspired aesthetic and robust engineering practices.

## Features

- **Real-time Classification**: Identifies 10 market apple varieties.
- **Confidence-Aware**: Filters low-confidence predictions to prevent errors.
- **Top-5 Probabilities**: Visualizes the model's uncertainty.
- **Modern UI**: Clean, responsive design with drag-and-drop functionality.
- **Performance**: Cached model loading for fast inference.

## Project Structure

```
apple-variety-streamlit/
├── app.py              # Main application entry point
├── model/
│   └── apple_classifier.keras  # Place your trained model here
├── assets/
│   └── styles.css      # Custom CSS for styling
├── labels.json         # Class index to name mapping
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Setup & Installation

### Prerequisities

- Python 3.9+
- [Optional] Docker

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Your Model

Place your trained Keras model file (`apple_classifier.keras`) inside the `model/` directory.

### 3. Run the App

```bash
streamlit run app.py
```

## Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in to [Streamlit Cloud](https://streamlit.io/cloud).
3. Connect your GitHub repository.
4. Select `app.py` as the main file.
5. Deploy!

### Docker

You can containerize this application for deployment on any cloud provider.

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

## Classes

The model is trained to recognize the following classes:

1. Apple 10
2. Apple 13
3. Apple 18
4. Apple 19
5. Apple 7
6. Apple 8
7. Apple 9
8. Apple Red Yellow 2
9. Apple hit 1
10. Apple worm 1

## License

MIT
