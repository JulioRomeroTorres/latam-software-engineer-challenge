import fastapi
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import joblib

app = fastapi.FastAPI()
model = None

class FlightFeatures(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: str 

class InputPrediction(BaseModel):
    flights: List[FlightFeatures]


def convert_dict_to_df(
    dict_features: Dict[str, Any]
) -> pd.DataFrame:
    data = pd.DataFrame(dict_features)
    features = pd.concat(
        [
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')
        ],
        axis = 1
    )
    return features

@app.on_event("startup")
def startup_event():
    artifact_path = "./challenge/artifacts"
    _model = joblib.load(f"{artifact_path}/model.joblib")
    model = _model

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: InputPrediction) -> dict:

    list_features = request.flights

    features_df = convert_dict_to_df(list_features)
    return {
        "predictions": model.predict(features_df).toList()
    }