from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from backend.services.prediction_service import predict_future, predict_next_day
from backend.services.sentiment_service import combined_prediction

router = APIRouter()


# ---------------------------
# Request Model
# ---------------------------

class FuturePredictionRequest(BaseModel):
    symbol: str
    days_ahead: int


# ---------------------------
# Next Day Prediction
# ---------------------------

@router.get("/predict/next-day/{symbol}")
def get_next_day_prediction(symbol: str):
    try:
        result = predict_next_day(symbol.upper())
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------
# Future Prediction
# ---------------------------

@router.post("/predict/future")
def get_future_prediction(request: FuturePredictionRequest):
    try:
        result = predict_future(
            request.symbol.upper(),
            request.days_ahead
        )
        return {"predictions": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------
# Combined Prediction (Price + Sentiment)
# ---------------------------

@router.get("/predict/combined/{symbol}")
def combined(symbol: str, days_ahead: int = Query(1, ge=1, le=14)):
    """
    Get combined price and sentiment prediction
    - symbol: Stock symbol (e.g., RELIANCE.NS)
    - days_ahead: Number of days to predict (1-14, default=1)
    """
    try:
        return combined_prediction(symbol.upper(), days_ahead)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))