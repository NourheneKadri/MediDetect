# Importer les bibliothèques nécessaires
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Charger le modèle sauvegardé
with open('classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Initialiser l'application Flask
app = Flask(__name__)

# Créer la route pour la racine '/'
@app.route('/')
def home():
    return "Bienvenue sur le serveur Flask!"

# Créer la route POST pour prédiction
# Créer la route POST pour prédiction
@app.route('/predict', methods=['POST'])
def predict():
    # Recevoir les données envoyées dans la requête POST en JSON
    data = request.get_json()

    # Extraire les données (assurons-nous que le client envoie bien les champs corrects)
    age = data['age']
    glucose = data['glucose']
    blood_pressure = data['blood_pressure']
    skin_thickness = data['skin_thickness']  
    insulin = data['insulin']
    bmi = data['bmi']
    diabetes_pedigree_function = data['diabetes_pedigree_function']
    pregnancies = data['pregnancies']

    # Préparer les données pour le modèle
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    # Faire la prédiction
    prediction = model.predict(input_data)

    # Convertir le type de prédiction pour qu'il soit JSON sérialisable
    prediction_value = int(prediction[0])  # Convertir en int

    # Retourner le résultat comme réponse JSON
    return jsonify({
        'prediction': prediction_value  # Renvoie le résultat de la prédiction
    })

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
