
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(
    title="Solar Power Forecasting API",
    description="Backend API for the solar power forecasting project",
    version="1.0.0",
)

# Load the trained model from this app folder
MODEL_PATH = Path(__file__).resolve().parent / "solar_model.joblib"
model = joblib.load(MODEL_PATH)

# The model expects these 8 features in this exact order
FEATURE_NAMES = [
    "GHI",
    "DNI",
    "DHI",
    "Temperature",
    "Humidity",
    "Pressure",
    "Wind Speed",
    "Solar Zenith Angle",
]

if getattr(model, "n_features_in_", None) != len(FEATURE_NAMES):
    raise RuntimeError("Model must accept exactly 8 input features.")


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ghi: float = Field(ge=0, allow_inf_nan=False)
    dni: float = Field(ge=0, allow_inf_nan=False)
    dhi: float = Field(ge=0, allow_inf_nan=False)
    temperature: float = Field(allow_inf_nan=False)
    humidity: float = Field(ge=0, le=100, allow_inf_nan=False)
    pressure: float = Field(gt=0, allow_inf_nan=False)
    wind_speed: float = Field(ge=0, allow_inf_nan=False)
    solar_zenith_angle: float = Field(
        ge=0, le=180, allow_inf_nan=False
    )


@app.get("/")
def home():
    return {
        "message": "Solar Power Forecasting API is running",
        "status": "success",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "solar-forecasting-api",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        features = np.array(
            [[
                data.ghi,
                data.dni,
                data.dhi,
                data.temperature,
                data.humidity,
                data.pressure,
                data.wind_speed,
                data.solar_zenith_angle,
            ]],
            dtype=float,
        )

        prediction = float(model.predict(features)[0])

        if not np.isfinite(prediction):
            raise HTTPException(
                status_code=500,
                detail="Model returned an invalid prediction.",
            )

        return {
            "status": "success",
            "prediction": prediction,
            "unit": "W/m²",
            "features_used": FEATURE_NAMES,
        }

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        )
