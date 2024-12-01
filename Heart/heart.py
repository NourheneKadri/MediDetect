from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware


# Charger le modèle sauvegardé
with open('Heart.pkl', 'rb') as file:
    model = pickle.load(file)


# Initialisation de FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# Schéma de validation pour les données d'entrée
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict/")
def predict(data: PatientData):
    # Conversion des données en format attendu par le modèle
    features = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])

    # Prédire le résultat avec le modèle
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features).max()  # Probabilité associée

    # Ajouter des messages explicatifs
    if prediction == 0:
        message = "The Person does not have a Heart Disease."
    else:
        message = "The Person has Heart Disease."

    return {
        "prediction": int(prediction),  # 0 ou 1
        "probability": probability,      # Confiance de la prédiction
        "message": message               # Message explicatif
    }