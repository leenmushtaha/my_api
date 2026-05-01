"""
main.py
-------
FastAPI server for Spiral Tremor Classification.
Receives spiral coordinates from Unity, extracts features,
runs the Random Forest model, and returns the prediction.

Setup:
    pip install fastapi uvicorn scikit-learn numpy scipy joblib pydantic

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

API Endpoints:
    GET  /          - Health check
    POST /predict   - Main prediction endpoint
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import joblib
import uvicorn


# =============================================================================
# FEATURE EXTRACTION (copied from feature_extraction.py)
# =============================================================================

def fit_ideal_spiral(coords):
    cx, cy = coords.mean(axis=0)
    dx = coords[:, 0] - cx
    dy = coords[:, 1] - cy
    radii = np.sqrt(dx**2 + dy**2)
    angles = np.arctan2(dy, dx)
    angles_unwrapped = np.unwrap(angles)
    A = np.vstack([np.ones_like(angles_unwrapped), angles_unwrapped]).T
    result = np.linalg.lstsq(A, radii, rcond=None)
    a, b = result[0]
    ideal_radii = a + b * angles_unwrapped
    ideal_x = cx + ideal_radii * np.cos(angles)
    ideal_y = cy + ideal_radii * np.sin(angles)
    return np.column_stack([ideal_x, ideal_y])


def extract_features(coords):
    features = {}
    coords = np.array(coords, dtype=float)

    if len(coords) < 10:
        return None

    # Deviation from ideal spiral
    try:
        ideal = fit_ideal_spiral(coords)
        deviations = np.sqrt(((coords - ideal) ** 2).sum(axis=1))
        features["deviation_mean"] = round(float(deviations.mean()), 6)
        features["deviation_std"]  = round(float(deviations.std()), 6)
        features["deviation_max"]  = round(float(deviations.max()), 6)
    except Exception:
        features["deviation_mean"] = 0.0
        features["deviation_std"]  = 0.0
        features["deviation_max"]  = 0.0

    # Jitter
    diffs = np.diff(coords, axis=0)
    step_sizes = np.sqrt((diffs ** 2).sum(axis=1))
    features["jitter_mean"] = round(float(step_sizes.mean()), 6)
    features["jitter_std"]  = round(float(step_sizes.std()), 6)
    features["jitter_max"]  = round(float(step_sizes.max()), 6)

    # Curvature
    if len(diffs) >= 2:
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        angle_changes = np.abs(np.diff(np.unwrap(angles)))
        features["curvature_mean"] = round(float(angle_changes.mean()), 6)
        features["curvature_std"]  = round(float(angle_changes.std()), 6)
    else:
        features["curvature_mean"] = 0.0
        features["curvature_std"]  = 0.0

    # Path length ratio
    try:
        actual_length = step_sizes.sum()
        ideal_diffs = np.diff(ideal, axis=0)
        ideal_length = np.sqrt((ideal_diffs ** 2).sum(axis=1)).sum()
        features["path_length_ratio"] = round(
            float(actual_length / (ideal_length + 1e-8)), 6)
    except Exception:
        features["path_length_ratio"] = 1.0

    # Radial std
    cx, cy = coords.mean(axis=0)
    radii = np.sqrt((coords[:, 0] - cx)**2 + (coords[:, 1] - cy)**2)
    features["radial_std"] = round(float(radii.std()), 6)

    return features


# =============================================================================
# FASTAPI APP SETUP
# =============================================================================

app = FastAPI(
    title="Spiral Tremor Classifier API",
    description="Classifies spiral drawings as Healthy or Parkinson's",
    version="1.0.0"
)

# CORS middleware — allows website and Unity to call the API from any origin
# In production, replace "*" with your actual website domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model once when server starts
# Make sure spiral_model.pkl is in the same folder as main.py
try:
    model = joblib.load("spiral_model.pkl")
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Model file not found — make sure spiral_model.pkl is present")
    model = None


# =============================================================================
# REQUEST AND RESPONSE MODELS
# =============================================================================

class SpiralPoint(BaseModel):
    """A single point from the Unity game session."""
    shipX: float
    shipY: float
    # handX and handY are optional — include if available from Unity
    handX: Optional[float] = None
    handY: Optional[float] = None
    time:  Optional[float] = None


class PredictRequest(BaseModel):
    """
    Request body sent from Unity or website.
    Contains the list of spiral points from one game session.
    """
    patient_id:   Optional[str] = None   # optional patient identifier
    session_id:   Optional[str] = None   # optional session identifier
    points: List[SpiralPoint]            # the spiral coordinate data


class PredictResponse(BaseModel):
    """
    Response returned to Unity and website.
    Contains prediction, confidence, and extracted features.
    """
    patient_id:     Optional[str]
    session_id:     Optional[str]
    prediction:     int           # 0 = Healthy, 1 = Parkinson's
    label:          str           # "Healthy" or "Parkinson's"
    confidence:     float         # probability of predicted class (0.0 - 1.0)
    features:       dict          # the 10 extracted features
    point_count:    int           # number of points received
    status:         str           # "success" or "error"


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
def health_check():
    """
    Health check endpoint.
    Visit this URL to confirm the server is running.
    """
    return {
        "status": "running",
        "model_loaded": model is not None,
        "message": "Spiral Tremor Classifier API is running"
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Main prediction endpoint.
    Receives spiral points, extracts features, returns classification.

    Unity sends a POST request to this endpoint with the spiral data.
    Website reads the response to display on the dashboard.
    """

    # Check model is loaded
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Make sure spiral_model.pkl exists."
        )

    # Check enough points were received
    if len(request.points) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough points received ({len(request.points)}). Minimum is 10."
        )

    # Extract shipX, shipY coordinates from the request
    coords = np.array([
        [point.shipX, point.shipY]
        for point in request.points
    ])

    # Extract features from coordinates
    features = extract_features(coords)

    if features is None:
        raise HTTPException(
            status_code=400,
            detail="Feature extraction failed — not enough valid points."
        )

    # Prepare feature vector in the correct order for the model
    feature_order = [
        "deviation_mean", "deviation_std", "deviation_max",
        "jitter_mean", "jitter_std", "jitter_max",
        "curvature_mean", "curvature_std",
        "path_length_ratio", "radial_std"
    ]
    feature_vector = np.array(
        [[features[f] for f in feature_order]]
    )

    # Run model prediction
    prediction = int(model.predict(feature_vector)[0])
    probabilities = model.predict_proba(feature_vector)[0]
    confidence = round(float(probabilities[prediction]), 4)
    label = "Parkinson's" if prediction == 1 else "Healthy"

    return PredictResponse(
        patient_id=request.patient_id,
        session_id=request.session_id,
        prediction=prediction,
        label=label,
        confidence=confidence,
        features=features,
        point_count=len(request.points),
        status="success"
    )


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
