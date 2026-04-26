from backend.services.prediction_service import predict_next_day
from backend.services.sentiment_service import get_sentiment_score


def predict_combined(symbol: str):

    # 1️⃣ Get stock prediction (next day)
    stock = predict_next_day(symbol)

    if not stock:
        return None

    # 2️⃣ Get sentiment score
    sentiment_score = get_sentiment_score(symbol)

    # 3️⃣ Decide final signal
    if sentiment_score > 0.2:
        final_signal = "BUY"
    elif sentiment_score < -0.2:
        final_signal = "SELL"
    else:
        final_signal = "HOLD"

    # 4️⃣ Return combined result
    return {
        "symbol": symbol,
        "prediction_date": stock["date"],
        "predicted_open": stock["open"],
        "predicted_high": stock["high"],
        "predicted_low": stock["low"],
        "predicted_close": stock["close"],
        "sentiment_score": sentiment_score,
        "final_signal": final_signal
    }
