from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from datetime import datetime, timedelta
import pandas as pd
import os
import traceback

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],                     
    allow_headers=["*"],                   
)
MODEL_PATH = 'new_model.pkl'
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file missing. Train first: {os.path.abspath(MODEL_PATH)}")
try:
    with open(MODEL_PATH, 'rb') as f:
        saved_data = pickle.load(f)
        
    if not isinstance(saved_data, dict):
        raise RuntimeError("Model file corrupted - expected dictionary")
        
    model = saved_data.get('model')
    encoders = saved_data.get('encoders', {})
    
    if not model or not encoders:
        raise RuntimeError("Model file missing components")

except Exception as e:
    raise RuntimeError(f"Model load error: {str(e)}")

class CropPricePredictor:
    def __init__(self, model, encoders):
        self.model = model
        self.encoders = encoders

    def _encode(self, feature_type: str, value: str):
        le = self.encoders.get(feature_type)
        if not le:
            raise ValueError(f"Missing encoder for {feature_type}")
        if value not in le.classes_:
            raise ValueError(f"Unknown {feature_type}: {value}")
        return le.transform([value])[0]

    def predict(self, input_data: dict):
        current_date = datetime.strptime(input_data['current_date'], "%Y-%m-%d")
        features = pd.DataFrame([{
            'state_code': self._encode('state', input_data['state']),
            'district_code': self._encode('district', input_data['district']),
            'market_code': self._encode('market', input_data['market']),
            'crop_code': self._encode('crop', input_data['crop']),
            'day_of_year': current_date.timetuple().tm_yday,
            'month': current_date.month,
            '3day_momentum': 0.0,
            'current_price': input_data['current_price']
        }])
        return round(self.model.predict(features)[0], 2)

class PredictionRequest(BaseModel):
    state: str
    district: str
    market: str
    crop: str
    current_date: str
    current_price: float

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        predictor = CropPricePredictor(model, encoders)
        prediction = predictor.predict(request.model_dump())
        target_date = datetime.strptime(request.current_date, "%Y-%m-%d") + timedelta(days=9)
        return {
            "predicted_price": f"₹{prediction}/kg",
            "prediction_date": target_date.strftime("%B"),
            "currency": "INR"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
