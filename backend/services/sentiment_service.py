import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from backend.services.prediction_service import predict_future

# =====================================================
# CONFIG
# =====================================================

NEWS_API_KEY = "c8684555fb964ae2b46096326df0cfc9"

MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

labels = ["Negative", "Neutral", "Positive"]


# =====================================================
# FETCH NEWS FROM NEWSAPI
# =====================================================

def fetch_news(symbol):
    try:
        # Symbol → Company Name Mapping
        symbol_map = {
            "RELIANCE.NS": "Reliance Industries",
            "TCS.NS": "Tata Consultancy Services",
            "INFY.NS": "Infosys",
            "HDFCBANK.NS": "HDFC Bank",
            "ICICIBANK.NS": "ICICI Bank",
            "SBIN.NS": "State Bank of India",
            "ITC.NS": "ITC Limited",
            "LT.NS": "Larsen & Toubro",
            "HINDUNILVR.NS": "Hindustan Unilever",
            "BAJFINANCE.NS": "Bajaj Finance"
        }

        company_name = symbol_map.get(symbol.upper(), symbol.replace('.NS', ''))

        url = (
            f"https://newsapi.org/v2/everything?"
            f"q=\"{company_name}\"&"
            f"language=en&"
            f"sortBy=publishedAt&"
            f"pageSize=10&"
            f"apiKey={NEWS_API_KEY}"
        )

        response = requests.get(url).json()

        articles = response.get("articles", [])

        news_list = []

        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")

            combined_text = title + " " + (description or "")
            news_list.append(combined_text)

        print(f"DEBUG → Company: {company_name}, News Count: {len(news_list)}")

        return news_list

    except Exception as e:
        print("News Fetch Error:", e)
        return []


# =====================================================
# FINBERT SENTIMENT PREDICTION
# =====================================================

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    return probs.squeeze().tolist()


# =====================================================
# CALCULATE SENTIMENT SCORE
# =====================================================

def sentiment_score(company):
    news_list = fetch_news(company)

    if not news_list:
        return 0.0

    scores = []

    for news in news_list:
        neg, neu, pos = predict_sentiment(news)
        score = pos - neg
        scores.append(score)

    return sum(scores) / len(scores)


# =====================================================
# GET SENTIMENT EXPLANATION
# =====================================================

def get_sentiment_explanation(sentiment_score, news_articles):
    """
    Generate detailed explanation of sentiment analysis
    """
    if not news_articles:
        return {
            "summary": "No recent news articles found for sentiment analysis.",
            "positive_factors": [],
            "negative_factors": [],
            "neutral_factors": ["No news data available"],
            "key_topics": [],
            "article_breakdown": {"positive": 0, "neutral": 0, "negative": 0, "total": 0}
        }
    
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    key_topics = []
    
    # Analyze top 5 articles for detailed explanation
    for article in news_articles[:5]:
        neg, neu, pos = predict_sentiment(article)
        
        if pos > 0.6:
            positive_count += 1
            # Extract key positive phrases
            article_lower = article.lower()
            if "profit" in article_lower or "earnings" in article_lower:
                key_topics.append("earnings growth")
            if "launch" in article_lower or "new" in article_lower:
                key_topics.append("product launch")
            if "expansion" in article_lower or "growth" in article_lower:
                key_topics.append("business expansion")
            if "dividend" in article_lower:
                key_topics.append("dividend announcement")
            if "partnership" in article_lower or "collaboration" in article_lower:
                key_topics.append("strategic partnership")
        elif neg > 0.6:
            negative_count += 1
            article_lower = article.lower()
            if "loss" in article_lower or "decline" in article_lower:
                key_topics.append("financial losses")
            if "regulatory" in article_lower or "investigation" in article_lower:
                key_topics.append("regulatory concerns")
            if "competition" in article_lower:
                key_topics.append("increased competition")
            if "downgrade" in article_lower:
                key_topics.append("analyst downgrade")
            if "lawsuit" in article_lower or "legal" in article_lower:
                key_topics.append("legal issues")
        else:
            neutral_count += 1
    
    unique_topics = list(set(key_topics))[:5]
    
    explanation = {
        "summary": "",
        "positive_factors": [],
        "negative_factors": [],
        "neutral_factors": [],
        "key_topics": unique_topics,
        "article_breakdown": {
            "positive": positive_count,
            "neutral": neutral_count,
            "negative": negative_count,
            "total": len(news_articles[:5])
        }
    }
    
    # Generate summary based on sentiment score
    if sentiment_score > 0.5:
        explanation["summary"] = "Strong positive sentiment driven by recent news and market optimism."
        explanation["positive_factors"] = [
            "Multiple positive news articles detected",
            "Favorable market sentiment",
            "Positive earnings expectations",
            "Strong business momentum"
        ]
        if not explanation["negative_factors"]:
            explanation["negative_factors"] = ["No significant negative news detected"]
            
    elif sentiment_score > 0.2:
        explanation["summary"] = "Moderately positive sentiment with some mixed signals."
        explanation["positive_factors"] = [
            "Some positive news coverage",
            "Generally optimistic market view",
            "Stable business outlook"
        ]
        explanation["negative_factors"] = ["Minor concerns present but not dominant"]
        
    elif sentiment_score > -0.2:
        explanation["summary"] = "Neutral sentiment - market is waiting for clearer direction."
        explanation["neutral_factors"] = [
            "Balanced news coverage",
            "No strong market-moving news",
            "Trading within expected range"
        ]
        
    elif sentiment_score > -0.5:
        explanation["summary"] = "Moderately negative sentiment detected."
        explanation["negative_factors"] = [
            "Some concerning news articles",
            "Cautious market outlook",
            "Potential headwinds detected"
        ]
        explanation["positive_factors"] = ["Some positive aspects but sentiment leaning negative"]
        
    else:
        explanation["summary"] = "Strong negative sentiment - multiple risk factors identified."
        explanation["negative_factors"] = [
            "Predominantly negative news coverage",
            "Market concerns about company outlook",
            "High uncertainty levels",
            "Multiple risk factors identified"
        ]
        explanation["positive_factors"] = ["Limited positive catalysts at this time"]
    
    return explanation


# =====================================================
# COMBINED PRICE + SENTIMENT ENGINE
# Supports Next Day + Future Days
# =====================================================

def combined_prediction(symbol, days_ahead=1):
    try:
        # Get price predictions
        price_predictions = predict_future(symbol, days_ahead)
        last_prediction = price_predictions[-1]
        
        # Get news for sentiment
        news_list = fetch_news(symbol)
        sentiment = sentiment_score(symbol)
        
        # Get detailed explanation
        explanation = get_sentiment_explanation(sentiment, news_list)
        
        # Determine signal
        price_trend = last_prediction["close"] - last_prediction["open"]
        
        if price_trend > 0 and sentiment > 0.2:
            final_signal = "Strong Buy"
            signal_strength = "High"
        elif price_trend > 0 and sentiment > 0:
            final_signal = "Buy"
            signal_strength = "Moderate"
        elif price_trend > 0 and sentiment < 0:
            final_signal = "Weak Buy"
            signal_strength = "Low"
        elif price_trend < 0 and sentiment < -0.2:
            final_signal = "Strong Sell"
            signal_strength = "High"
        elif price_trend < 0 and sentiment < 0:
            final_signal = "Sell"
            signal_strength = "Moderate"
        elif price_trend < 0 and sentiment > 0:
            final_signal = "Weak Sell"
            signal_strength = "Low"
        else:
            final_signal = "Neutral"
            signal_strength = "Neutral"
        
        return {
            "symbol": symbol,
            "prediction_date": last_prediction["date"],
            "predicted_open": last_prediction["open"],
            "predicted_close": last_prediction["close"],
            "predicted_high": last_prediction["high"],
            "predicted_low": last_prediction["low"],
            "sentiment_score": round(sentiment, 3),
            "sentiment_explanation": explanation,
            "final_signal": final_signal,
            "signal_strength": signal_strength,
            "price_trend": "Up" if price_trend > 0 else "Down",
            "news_analyzed": len(news_list)
        }
        
    except Exception as e:
        return {"error": str(e)}