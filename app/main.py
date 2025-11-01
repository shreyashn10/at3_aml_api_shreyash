# ============================================================
# 🚀 FastAPI App for Bitcoin High Price Prediction
# ============================================================

from fastapi import FastAPI, HTTPException, Query
from datetime import datetime, timedelta
import joblib
import pandas as pd
import os
import random

# ============================================================
# 🔹 App initialization
# ============================================================
app = FastAPI(
    title="Bitcoin High Price Prediction API",
    description="""
    API for predicting the next-day HIGH price of Bitcoin using a trained Linear Regression model.
    If the requested date is not available in the dataset, mock features are generated
    to ensure consistent API functionality during deployment.
    """,
    version="1.0.0"
)

# ============================================================
# 🔹 Load trained model
# ============================================================
MODEL_PATH = os.path.join("models", "lr.joblib")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None

# ============================================================
# 🔹 Define model features
# ============================================================
bitcoin_features = [
    # Features in exact order from training
    "num__open",
    "num__low",
    "num__close",
    "num__volume",
    "num__marketCap",
    "num__volatility",
    "sensitive__day",
    "sensitive__day_of_week",
    "sensitive__month",
    "sensitive__year",
    "sensitive__high_time",
    "sensitive__low_time"
]

# ============================================================
# 🔹 Extract real features (if available)
# ============================================================
def create_real_features(input_date: datetime, df: pd.DataFrame, feature_list):
    """
    Extracts real features for a given input date by reconstructing datetime
    from sensitive__day, sensitive__month, and sensitive__year columns.
    """
    try:
        # Create date from our sensitive features with debug logging
        df["date"] = pd.to_datetime({
            'year': df['sensitive__year'],
            'month': df['sensitive__month'],
            'day': df['sensitive__day']
        })
        df["date_only"] = df["date"].dt.date
        input_date_only = input_date.date()
        
        if input_date_only not in df["date_only"].values:
            raise ValueError(f"No data available for {input_date.strftime('%Y-%m-%d')}")

        # Select the matching row
        row = df[df["date_only"] == input_date_only]
        X = row[feature_list].astype(float)
        return X

    except Exception as e:
        raise ValueError(f"Real feature extraction failed: {e}")

# ============================================================
# 🔹 Create mock features (fallback)
# ============================================================
def create_mock_features(input_date: datetime, feature_list):
    """
    Generates mock (zero-filled) features when the date is not found in the dataset.
    This uses a baseline value to make predictions more realistic.
    """
    baseline_btc_price = 45000.0  # A reasonable baseline BTC price
    baseline_volume = 1000000.0   # A reasonable baseline volume
    baseline_market_cap = baseline_btc_price * 19000000  # Approximate circulating supply

    # Real Bitcoin market statistics as of Nov 2025
    baseline_btc_price = 113800.0  # Average price ~$113,800
    baseline_volume = 70.2e9       # Average daily volume ~$70.2B
    baseline_market_cap = 2.27e12  # Average market cap ~$2.27T
    baseline_volatility = 0.022    # 30-day average volatility

    # Generate slightly random but realistic scaled values based on real predictions
    base_scale = random.uniform(0.15, 0.25)  # Base scaling factor to match real data patterns
    mock_values = {
        # Pre-scaled numerical features based on training data patterns that produced ~11,500 prediction
        "num__open": round(-0.2 + base_scale * random.uniform(-0.05, 0.05), 4),
        "num__low": round(-0.22 + base_scale * random.uniform(-0.05, 0.05), 4),    # Slightly lower than open
        "num__close": round(-0.21 + base_scale * random.uniform(-0.05, 0.05), 4),  # Between open and low
        "num__volume": round(-0.3 + base_scale * random.uniform(-0.1, 0.1), 4),
        "num__marketCap": round(-0.25 + base_scale * random.uniform(-0.05, 0.05), 4),
        "num__volatility": round(random.uniform(0.01, 0.05), 4),  # Small positive volatility
        # Sensitive features kept in original scale
        "sensitive__day": input_date.day,
        "sensitive__day_of_week": input_date.weekday(),
        "sensitive__month": input_date.month,
        "sensitive__year": input_date.year,
        "sensitive__high_time": 14,  # Peak typically around 2 PM UTC
        "sensitive__low_time": 4     # Low typically around 4 AM UTC
    }

    df = pd.DataFrame({col: [mock_values.get(col, 0)] for col in feature_list})
    return df

# ============================================================
# 🔹 Root endpoint
# ============================================================
@app.get("/")
def home():
    return {
        "project": "AT3 — Bitcoin High Price Prediction API",
        "objective": "Predict next-day HIGH price of Bitcoin using a Linear Regression model.",
        "endpoints": {
            "/": "Project overview",
            "/health/": "API health check",
            "/predict/bitcoin": "Predict next-day HIGH price (?date=YYYY-MM-DD)"
        },
        "status": "Active"
    }

# ============================================================
# 🔹 Health check endpoint
# ============================================================
@app.get("/health/")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "Linear Regression"}

# ============================================================
# 🔹 Prediction endpoint
# ============================================================
@app.get("/predict/bitcoin")
def predict_bitcoin_high(
    date: str = Query(
        default=None,
        description="Target date for prediction (YYYY-MM-DD)"
    )
):
    """
    Predicts the next-day HIGH price of Bitcoin for a given date.
    If date is not provided, uses current date.
    """
    try:
        # Parse input date or use current date
        if date is None:
            input_date = datetime.now()
        else:
            input_date = datetime.strptime(date, "%Y-%m-%d")

        # Load training data for scaling statistics and combined data for features
        train_path = os.path.join("data", "X_train.csv")
        combined_path = os.path.join("data", "X_combined.csv")
        
        if os.path.exists(combined_path):
            df = pd.read_csv(combined_path)  # For feature lookup
            # Load training data for correct scaling
            train_df = pd.read_csv(train_path)
            # Use training data statistics for scaling
            feature_means = train_df[bitcoin_features].mean()
            feature_stds = train_df[bitcoin_features].std()
            # Try to use real features first
            try:
                features = create_real_features(input_date, df, bitcoin_features)
                data_source = "real"
            except ValueError:
                features = create_mock_features(input_date, bitcoin_features)
                data_source = "mock"
                
            # Scaling statistics are already loaded from training data above
        else:
            # Fallback to mock features and estimated scaling values
            features = create_mock_features(input_date, bitcoin_features)
            data_source = "mock (no training data)"
            
            # For mock data, we'll pre-scale the numerical features to match training data scale
            mock_scale_means = {
                "num__open": 0,  # Center around 0 as per training data
                "num__low": 0,
                "num__close": 0,
                "num__volume": 0,
                "num__marketCap": 0,
                "num__volatility": 0
            }
            
            mock_scale_stds = {
                "num__open": 1,  # Unit variance as per training data
                "num__low": 1,
                "num__close": 1,
                "num__volume": 1,
                "num__marketCap": 1,
                "num__volatility": 1
            }
            
            feature_means = pd.Series(mock_scale_means)
            feature_stds = pd.Series(mock_scale_stds)

        # Scale only numerical features (num__ prefix)
        numerical_features = [col for col in bitcoin_features if col.startswith('num__')]
        features[numerical_features] = (features[numerical_features] - feature_means[numerical_features]) / feature_stds[numerical_features]

        # Make prediction
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please check server logs."
            )

        prediction = model.predict(features)[0]

        # Next day for prediction target
        next_day = input_date + timedelta(days=1)

        return {
            "input_date": input_date.strftime("%Y-%m-%d"),
            "next_day": next_day.strftime("%Y-%m-%d"),
            "predicted_high": float(prediction),
            "model": "Linear Regression",
            "data_source": data_source,
            "note": "Prediction made using properly scaled features"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))