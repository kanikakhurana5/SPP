from fastapi import APIRouter, Query
from backend.services.sentiment_service import combined_prediction

router = APIRouter()


@router.get("/predict/combined/{symbol}")
def combined(symbol: str, days_ahead: int = Query(1, ge=1, le=14)):

    return combined_prediction(symbol.upper(), days_ahead)
