# Bitcoin Price Prediction API by Shreyash Narayane

## 📘 Overview

This FastAPI## 🔧 Technical Details
- Python 3.11
- FastAPI framework
- Linear Regression model (scikit-learn)
- Data preprocessing with pandas
- Docker containerization

## 🧩 Integration
This API is part of the Group 3 Cryptocurrency Forecast project, designed to provide reliable Bitcoin price predictions through a clean, modern FASTAPI interface.cation predicts the next-day HIGH price of Bitcoin (BTC) based on historical market data. It is part of the Group 3 Cryptocurrency Forecast Dashboard developed for Advanced Machine Learning Applications (AT3) at UTS.

## 🚀 Features
- Prediction of next-day Bitcoin HIGH price
- Support for both historical and future dates
- Automatic fallback to mock data for unavailable dates
- Proper feature scaling based on training data
- Docker support for easy deployment

## 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Displays project overview, endpoints, and expected input/output |
| `/health` | GET | Returns a 200 status confirming the API is running |
| `/predict/bitcoin?date=<YYYY-MM-DD>` | GET | Returns predicted next-day HIGH price for Bitcoin |

### Example Usage
```
http://localhost:8000/predict/bitcoin?date=2020-10-18
```

### Example Response
```json
{
    "input_date": "2020-10-18",
    "next_day": "2020-10-19",
    "predicted_high": 11523.07,
    "model": "Linear Regression",
    "data_source": "real",
    "note": "Prediction made using properly scaled features"
}
```

## ⚙️ Setup and Installation

### Local Development
First, clone the repository:

```bash
git clone <https://github.com/shreyashn10/at3_aml_api_shreyash>
cd fastapi-at3
```

1. Create and activate Python virtual environment:
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API:
```bash
uvicorn app.main:app --reload
```

### Docker Deployment
1. Build the Docker image:
```bash
docker build -t bitcoin-prediction-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 bitcoin-prediction-api
```

## API Documentation
Once running, access the interactive API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🧠 Model Information
- **Algorithm:** Linear Regression
- **Framework:** scikit-learn
- **Features:** Historical price data and market indicators
- **Training Data:** Bitcoin price history (extensive dataset)
- **Prediction Target:** Next-day HIGH price
- **Scaling:** StandardScaler on numerical features only
- **Available Range:** Supports historical dates (with real data) and future dates (with mock data)

## 🔧 Technical Details
- Python 3.11
- FastAPI framework
- Linear Regression model (scikit-learn)
- Data preprocessing with pandas
- Docker containerization
