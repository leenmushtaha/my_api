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
import os
import json
from datetime import datetime, timezone
import firebase_admin
from firebase_admin import credentials, firestore


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
# FIREBASE SETUP
# =============================================================================

firebase_db = None

try:
    firebase_json = os.getenv({
  "type": "service_account",
  "project_id": "therapy-dashboard-f4ca9",
  "private_key_id": "28af4b10d23495ab9050e1833e6365e30ead86f0",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQC9Xt+AuwpTYmsK\nZfiWYgbvjxbOzKxk6qFPqS8L/HmFZbl82yi71EjPjw7EnnyPvjAkcBv5Jaj9bqD1\nSpbeSzc/jNgI65ZF96aU4Ss5gMWEYt5C1F8DLMap+EzxdvgIINtGs1RPi/qLnyK9\nRWP/ho91gv5hPbRd6GCWQqKhw1RKYEhN1UxDw9srA7zxYLIdIi6IIS+40goBp9CT\nFUpbUudPuYFFvoxhAaZy7s5KeDf+HWesZ3pABe1mdyYJb0KDm73MhBjYDMHSDU3s\nsYrQVQExPcidfZz595MxXNHOWlnVjHNhqWZMzoVygEKMmEz5c5fD1RT1JjtvlwlQ\ndxUh4GlbAgMBAAECggEABkvlw0XHGZ1h5+PRqKgdUjVgH2hMDtYWDJDPT93t1weR\nEGA19pX7TqQBIzN71hG0BoNG52cMzZtEkKo2Kvq1+MbAurTqCbY9E30vuBk/E3KU\nsJcp05T76J1NtH5/z78XW2MWoqLApfX0GhrUef8zKqMqBflDeDuWQyHlWkiS29Xj\nKY6v74wkTNGiltbx0DjOb4qWmJvVkhurhnKxsaCkKTttS+MOXuTyUZrwuTSprdeo\npK2xBhXkG9EEDvaecNPTJV7tU2Ka0yjWTpU4BTrYBFYLWZqSysluVG5qXj9cGrZJ\nBPHHBCoSBeR6qcXsd86IjM7uq0DKngmrU5cj6gprkQKBgQDw2pYiiY3bPNcnTEe0\nmt4WkWUriuzZUCyssABAkdwgIoRlKCMQVq3qBjJQljUChnU64NDk5LfRHcyhEysP\nJdhhiVPSFmq/hJATr2dqQn/JekEIE+Jsqw07qLGZXGn5DXZlf1izlRRH4xs6CgZe\nzLPWOGbmh/bhrh/eeMdby7bTkQKBgQDJR3o7UEi7ENjdYw+Qvl5mNz7fz/52DlJX\nfUi65wnMDNblGPnM0qmW++6oqxh3tZ0gAklodousSp11yL8sPWH0bZLt3Kc+7O8j\nTEpiMKtfq7xIi+5nHwyVF+qxOu5UxfWTuhfosLxO1ZBz6r+4P79NjO8+3EjvTeFx\n8EK8AkzgKwKBgQDNN3H0u39C7fPkZ/owyEOytu+cyiJEhyuJd+y/F4iXWNG13x0B\nLtnALMdyIonIPQhlwmg6nyZ/5wQTumFV5skXUgs5ViBeTnT0UN+sijyXTrNaTpb+\nQEBmNLYeFb+1lOLsWDUbzkoZdkgci64h2Aji3evPQMn6QIKm7AHxFQISAQKBgQCs\n+xvmS8ol0oW+RftDlwfD6ujDKpry1L4ZaJeP4S0/Sy2IOJ2+VLHhC2UBWgGuJ8wA\njVaPS4ogKQQIDN2XZK2BhoYGnGKzpqaifFdU6aTulMY8xt29jCahH6vYYuAexP6X\n1g/kL7e2PL5nkLDx5P9A48VdDa4004bUB/siXwu4fwKBgQCt5d3d7TVbtJuIfxLa\nPMiPZUOaFugl9j6rYVhOa8JiTVeDwtiWnhczzN+GccVH1C8d3AVnsQKkR++0L/v9\nAV2gZLAJc4+w5XsZsggaK4KOg75P6iYVL4ST33En4BlKi37tqZjEJlBo5/EUQAa+\n3t1JgpCzbRP18+D5o+4c5horuw==\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-fbsvc@therapy-dashboard-f4ca9.iam.gserviceaccount.com",
  "client_id": "101528367506146010849",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40therapy-dashboard-f4ca9.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
)
    firebase_project_id = os.getenv("therapy-dashboard-f4ca9")

    if firebase_json:
        firebase_credentials_dict = json.loads(firebase_json)
        cred = credentials.Certificate(firebase_credentials_dict)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                "projectId": firebase_project_id
            })

        firebase_db = firestore.client()
        print("✅ Firebase connected successfully")
    else:
        print("⚠️ FIREBASE_SERVICE_ACCOUNT_JSON not found. Firebase saving disabled.")

except Exception as e:
    print(f"❌ Firebase initialization failed: {e}")
    firebase_db = None

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
    firebase_saved: bool


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

    def save_prediction_to_firebase(result: dict):
        """
        Saves prediction result to Firebase Firestore.
        Collection: sessions
        Document ID: session_id
        """

        if firebase_db is None:
            print("⚠️ Firebase is not connected. Skipping save.")
            return False

        session_id = result.get("session_id")

        if not session_id:
            print("⚠️ Missing session_id. Cannot save to Firebase.")
            return False

        session_data = {
            "sessionId": result.get("session_id"),
            "patientId": result.get("patient_id"),
            "therapyType": "physical",
            "prediction": result.get("prediction"),
            "label": result.get("label"),
            "confidence": result.get("confidence"),
            "features": result.get("features"),
            "pointCount": result.get("point_count"),
            "status": result.get("status"),
            "timestamp": firestore.SERVER_TIMESTAMP,
        }

        firebase_db.collection("sessions").document(session_id).set(session_data)

        firebase_db.collection("commands").document(session_id).set({
            "status": "completed",
            "updatedAt": firestore.SERVER_TIMESTAMP
        }, merge=True)

        print(f"✅ Prediction saved to Firebase for session: {session_id}")
        return True

    result = {
        "patient_id": request.patient_id,
        "session_id": request.session_id,
        "prediction": prediction,
        "label": label,
        "confidence": confidence,
        "features": features,
        "point_count": len(request.points),
        "status": "success"
    }

    firebase_saved = save_prediction_to_firebase(result)

    result["firebase_saved"] = firebase_saved

    return result


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
