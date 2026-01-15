import fastapi
from pydantic import BaseModel, conint, Field
from typing import List, Dict, Any
import pandas as pd
import joblib
import os
from enum import Enum
from typing import Annotated

app = fastapi.FastAPI()
catalog = {}

class FlightValueEnum(Enum):
    NATIONAL = 'N'
    INTERNATIONAL = 'I'

class CompanyEnum(Enum):
    AMERICAN = "American Wings"
    LATAM = "Grupo LATAM"
    SKY = "Sky Airline"
    COPA = "Copa Air"
    ARGENTINAS = "Aerolineas Argentinas"

class FlightFeatures(BaseModel):
    OPERA: CompanyEnum
    TIPOVUELO: FlightValueEnum
    MES: Annotated[int, Field(ge=1, le=10)] 

class InputPrediction(BaseModel):
    flights: List[FlightFeatures]

def create_feature(additional_information: Dict[str, Any]):
    feature = {}
    print(f"additional_information {additional_information}")
    
    feature_name_list = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    for feature_name in feature_name_list:
        feature[feature_name] = 0
        if feature_name in additional_information:
            feature[feature_name] = additional_information[feature_name]
    return feature

def convert_list_to_df(
    features_list: List[Dict[str, Any]]
) -> pd.DataFrame:
    data = pd.DataFrame(features_list)
    initial_features_df = pd.concat(
        [
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')
        ],
        axis = 1
    )

    dict_features = initial_features_df.to_dict('records')

    converted_features = [ create_feature(dict_feature) for dict_feature in dict_features ]
    features_df = pd.DataFrame(converted_features) 

    print("Converted Features \n", features_df) 
    return features_df

def load_model():
    is_local_deployment = int(os.environ.get('IS_LOCAL_DEPLOYMENT', "0"))
    artifact_path = './artifacts' if is_local_deployment else "./challenge/artifacts"

    _model = joblib.load(f"{artifact_path}/model.joblib")
    catalog["model"] = _model

@app.on_event("startup")
def startup_event():
    load_model()
    
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: InputPrediction) -> dict:

    list_features = request.flights
    features_df = [ feature.dict() for feature in list_features] 
    features_df = convert_list_to_df(features_df)

    if "model" not in catalog:
        load_model()

    predictions = catalog["model"].predict(features_df)

    return {
        "predict": predictions.tolist()
    }