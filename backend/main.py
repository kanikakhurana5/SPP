'''
from fastapi import FastAPI
from backend.routes.prediction_routes import router as prediction_router
from backend.routes.stock_routes import router as stock_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Stock Prediction API")

# ---------------------------
# CORS (Frontend Connection)
# ---------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Include Routes
# ---------------------------

app.include_router(prediction_router)
app.include_router(stock_router)


@app.get("/")
def home():
    return {"message": "Stock Prediction API Running"}
'''

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.prediction_routes import router as prediction_router
from backend.routes.stock_routes import router as stock_router
from backend.routes.sentiment_routes import router as sentiment_router



app = FastAPI(title="Stock Prediction API")

# ---------------------------
# CORS FIX
# ---------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# ROUTES
# ---------------------------

app.include_router(prediction_router)
app.include_router(stock_router)
app.include_router(sentiment_router)

@app.get("/")
def home():
    return {"message": "API Running"}
