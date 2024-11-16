import re
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from math import isnan
from joblib import load

app = FastAPI()

FEATURE_NAMES = ['km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm', 'year']
CAT_FEATURES = ['seats', 'brand', 'model', 'seller_type', 'transmission', 'owner', 'fuel']

best_model = load('models/ridge.joblib')
onehot = load('models/onehot.joblib')

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]
    
def converte_mileage(mil):
    if mil is None or (isinstance(mil, float) and isnan(mil)): return None
    if mil.split(' ')[1] == 'kmpl':
        return float(mil.split(' ')[0]) / 1.40
    return float(mil.split(' ')[0])

def converte_engine(eng):
    if eng is None or (isinstance(eng, float) and isnan(eng)): return None
    return float(eng.split(' ')[0])

def converte_max_power(mp):
    if (
        mp is None or 
        (isinstance(mp, str) and mp.split(' ')[0] == '') or
        (isinstance(mp, float) and isnan(mp))
    ): return None
    return float(mp.split(' ')[0])

def extract_torque_rpm(entry):
    if entry is None or entry == '' or (isinstance(entry, float) and isnan(entry)): return None, None
    pattern = re.compile(
        r"""
        (?P<torque_value>\d+(?:\.\d+)?)
        \s*
        (?:Nm|nm|kgm)?
        \s*
        (?:@|at)?
        \s*
        (?P<rpm_range>[\d,]+(?:-[\d,]+)?)
        """,
        re.VERBOSE | re.IGNORECASE
    )
    match = pattern.search(entry)

    torque_value = match.group('torque_value')
    rpm_values = match.group('rpm_range')
    
    
    rpm_values = rpm_values.replace(',', '')
    
    if '-' in rpm_values:
        rpm_values = rpm_values.split('-')
        rpm_values = rpm_values[-1]
        
    if float(torque_value) <= 40: # оцениваю по порядку чтоб привести к nm, потому что где-то есть пропуски kgm или nm, так что плевать
        torque_value = float(torque_value)*9.8
    
    return torque_value, rpm_values

def prepare_data(item: Item) -> dict:
    torque, max_torque_rpm = extract_torque_rpm(item.torque)
    return {
        'brand': item.name.split(' ')[0],
        'model': item.name.split(' ')[1],
        'mileage': float(converte_mileage(item.mileage)),
        'engine': float(converte_mileage(item.engine)),
        'max_power': float(converte_max_power(item.max_power)),
        'torque': float(torque),
        'max_torque_rpm': float(max_torque_rpm),
        'km_driven': float(item.km_driven),
        'year': float(item.year),
        'seats': int(item.seats),
        'seller_type': item.seller_type,
        'transmission': item.transmission,
        'owner': item.owner,
        'fuel': item.fuel
    }

def predict_model(data: dict) -> float:
    one_hot_cat_features = onehot.transform([[data[cat_feature] for cat_feature in CAT_FEATURES]]).toarray()
    float_features = np.array([[data[float_feature] for float_feature in FEATURE_NAMES]])
    features_full = np.concatenate((float_features, one_hot_cat_features), axis=1)
    return best_model.predict(features_full)[0]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = prepare_data(item)
    print(data)
    predict_ = predict_model(data)
    return predict_


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    preds = []
    for item in items:
        data = prepare_data(item)
        predict_ = predict_model(data)
        preds += [predict_]
    return preds
