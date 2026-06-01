# Spiral Tremor Classifier — Random Forest Model

A machine learning API for classifying Parkinson's disease based on hand-drawn spiral coordinate data. This model is part of the VR-OT Connect system, a VR-based occupational therapy platform that uses AI to assist therapists in monitoring and evaluating Parkinson's patients.

---

## Overview

This repository contains the trained Random Forest classification model and the FastAPI server that serves it. The model receives spiral coordinate data recorded during a VR therapy session, extracts ten tremor-related features from the coordinates, and returns a binary classification of either Healthy or Parkinson's along with a confidence score.

---

## How It Works

1. The Unity VR game records the patient's hand movement coordinates during the spiral tracing exercise
2. The coordinates are sent to this API via an HTTP POST request
3. The API extracts ten features from the coordinates including deviation, jitter, curvature, path length ratio, and radial spread
4. The trained Random Forest model classifies the spiral as Healthy (0) or Parkinson's (1)
5. The result is returned to Unity, which forwards it to the Firebase database
6. The therapist dashboard reads the result from Firebase and displays it

---

## Repository Structure

```
├── main.py                  # FastAPI server and feature extraction logic
├── spiral_model.pkl         # Trained Random Forest model (serialized)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Model Details

| Property | Value |
|---|---|
| Model Type | Random Forest Classifier |
| Number of Trees | 200 |
| Max Depth | 5 |
| Class Weight | Balanced |
| Training Samples | 244 (122 per class) |
| Test Accuracy | 81.6% |
| AUC | 0.933 |
| Sensitivity | 0.83 |
| Specificity | 0.80 |

---

## Features Extracted

The following ten features are extracted from the spiral coordinates before classification:

| Feature | Description |
|---|---|
| `deviation_mean` | Average distance of points from the fitted ideal spiral |
| `deviation_std` | Standard deviation of deviation from the ideal spiral |
| `deviation_max` | Maximum deviation from the ideal spiral |
| `jitter_mean` | Average step size between consecutive points |
| `jitter_std` | Standard deviation of step sizes |
| `jitter_max` | Maximum step size between consecutive points |
| `curvature_mean` | Average angular change between consecutive path segments |
| `curvature_std` | Standard deviation of angular changes |
| `path_length_ratio` | Ratio of actual path length to ideal spiral path length |
| `radial_std` | Standard deviation of point distances from spiral center |

---

## API Endpoints

### GET /
Health check endpoint. Returns the server status and whether the model is loaded.

**Response:**
```json
{
  "status": "running",
  "model_loaded": true,
  "message": "Spiral Tremor Classifier API is running"
}
```

---

### POST /predict
Main prediction endpoint. Receives spiral coordinate data and returns a classification result.

**Request Body:**
```json
{
  "patient_id": "P001",
  "session_id": "S001",
  "points": [
    { "shipX": -8.3, "shipY": 0.24, "handX": -1.04, "handY": 1.28, "time": 0.0 },
    { "shipX": -8.1, "shipY": 0.31, "handX": -1.12, "handY": 1.35, "time": 0.02 }
  ]
}
```

**Response:**
```json
{
  "patient_id": "P001",
  "session_id": "S001",
  "prediction": 1,
  "label": "Parkinson's",
  "confidence": 0.77,
  "features": {
    "deviation_mean": 54.3,
    "deviation_std": 33.9,
    "deviation_max": 98.4,
    "jitter_mean": 312.4,
    "jitter_std": 12.1,
    "jitter_max": 45.2,
    "curvature_mean": 1.26,
    "curvature_std": 0.31,
    "path_length_ratio": 1.05,
    "radial_std": 48.2
  },
  "point_count": 500,
  "status": "success"
}
```

---

## Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/spiral-tremor-classifier.git
cd spiral-tremor-classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the server**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**4. Test the health check**

Open your browser and navigate to:
```
http://localhost:8000
```

**5. Access the interactive API documentation**

FastAPI provides automatic interactive documentation at:
```
http://localhost:8000/docs
```

---

## Deployment

This API is deployed on Railway. Any push to the main branch of this repository will trigger an automatic redeployment.

**Required files for Railway deployment:**

`requirements.txt`
```
fastapi
uvicorn
scikit-learn
numpy
scipy
joblib
pydantic
```

`Procfile`
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Integration

This API is designed to work as part of the VR-OT Connect system. It communicates with:

- **Unity VR Game** — receives coordinate data via POST request after each physical therapy session
- **Firebase Firestore** — results are forwarded to Firestore by Unity after the API responds
- **React Dashboard** — therapist dashboard reads results from Firestore and displays them

---

## Dataset

The model was trained on spiral coordinate data collected and preprocessed from multiple sources:

- Publicly available Parkinson's spiral image datasets
- UCI Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet dataset
- Images were processed using OpenCV-based coordinate extraction and skeletonization

---

## Limitations

- The model was trained on image-extracted coordinates and has not yet been validated on real VR session data
- Dataset size is limited to 244 samples which may affect generalization to unseen patient profiles
- Feature values extracted from Unity world coordinates operate on a different scale than the training data, which may affect prediction reliability until the model is retrained on real session data

---

## Future Work

- Retrain the model using coordinate data collected directly from real Unity VR sessions
- Expand the dataset to include a larger and more diverse patient population
- Extend classification from binary to multi-class tremor severity grading
- Conduct formal clinical validation trials with licensed occupational therapists

---

## Authors

Developed as part of a graduation project in Data Science.

---

## License

This project is intended for academic and research purposes only and is not approved for clinical diagnostic use.
