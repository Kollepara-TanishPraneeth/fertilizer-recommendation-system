# =============================================================
# app.py
# Fertilizer Recommendation System - Flask Web Application
# B.Tech 3rd Year ML Project | LPU | CSM355
# =============================================================

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

# ----------------------------------------------------------
# INITIALIZE FLASK APPLICATION
# ----------------------------------------------------------
app = Flask(__name__)

# ----------------------------------------------------------
# LOAD MODEL AND ENCODERS
# ----------------------------------------------------------
MODEL_PATH   = os.path.join('model', 'fertilizer_model.pkl')
ENCODER_PATH = os.path.join('model', 'encoder.pkl')

def load_model_and_encoders():
    """Load the trained model and label encoders from disk."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders
    except FileNotFoundError:
        return None, None

model, encoders = load_model_and_encoders()

# ----------------------------------------------------------
# VALID OPTIONS FOR DROPDOWNS
# ----------------------------------------------------------
SOIL_TYPES = ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
CROP_TYPES = ['Cotton', 'Maize', 'Paddy', 'Sugarcane', 'Tobacco', 'Wheat']

# Fertilizer details for result page
FERTILIZER_INFO = {
    'Urea': {
        'description': 'Urea is a nitrogen-rich fertilizer (46% N). It is widely used for all types of crops to boost green leaf growth and increase yield.',
        'usage': 'Apply before sowing or as top-dressing. Mix well into soil.',
        'color': '#27ae60'
    },
    'DAP': {
        'description': 'DAP (Di-Ammonium Phosphate) contains 18% Nitrogen and 46% Phosphate. It promotes strong root development and early plant growth.',
        'usage': 'Apply at the time of sowing. Suitable for most soil types.',
        'color': '#2980b9'
    },
    '14-35-14': {
        'description': 'A balanced NPK fertilizer with 14% Nitrogen, 35% Phosphorus, and 14% Potassium. Ideal for crops needing high phosphorus.',
        'usage': 'Apply during planting stage for best root and fruit development.',
        'color': '#8e44ad'
    },
    '17-17-17': {
        'description': 'A perfectly balanced NPK fertilizer with equal parts (17%) of Nitrogen, Phosphorus, and Potassium. Good for general crop nutrition.',
        'usage': 'Apply at any growth stage. Suitable for all crops and soils.',
        'color': '#e67e22'
    },
    '20-20': {
        'description': 'Contains 20% Nitrogen and 20% Phosphorus. Encourages vegetative growth and strong root system formation.',
        'usage': 'Ideal as a starter fertilizer applied at the time of sowing.',
        'color': '#16a085'
    },
    '28-28': {
        'description': 'High-concentration NPK fertilizer with 28% Nitrogen and 28% Phosphorus. Suitable for nitrogen and phosphorus deficient soils.',
        'usage': 'Apply carefully in measured doses to avoid over-fertilization.',
        'color': '#c0392b'
    },
    'Super Phosphate': {
        'description': 'Contains 16% Phosphorus. It improves soil fertility, root development, and helps crops absorb other nutrients effectively.',
        'usage': 'Best applied at the time of land preparation or sowing.',
        'color': '#d35400'
    }
}

# ----------------------------------------------------------
# ROUTE: HOME PAGE
# ----------------------------------------------------------
@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

# ----------------------------------------------------------
# ROUTE: ABOUT PAGE
# ----------------------------------------------------------
@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

# ----------------------------------------------------------
# ROUTE: PREDICTION PAGE (GET + POST)
# ----------------------------------------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle the prediction form — GET shows form, POST processes it."""

    # If model files are missing, show an error
    if model is None or encoders is None:
        error = "⚠️ Model files not found! Please run train_model.py first."
        return render_template('predict.html',
                               soil_types=SOIL_TYPES,
                               crop_types=CROP_TYPES,
                               error=error)

    if request.method == 'GET':
        # Just show the empty form
        return render_template('predict.html',
                               soil_types=SOIL_TYPES,
                               crop_types=CROP_TYPES,
                               error=None)

    # ---- POST: Process the form submission ----
    try:
        # Retrieve form values
        soil_type   = request.form.get('soil_type', '').strip()
        crop_type   = request.form.get('crop_type', '').strip()
        temperature = request.form.get('temperature', '').strip()
        humidity    = request.form.get('humidity', '').strip()
        moisture    = request.form.get('moisture', '').strip()
        nitrogen    = request.form.get('nitrogen', '').strip()
        phosphorous = request.form.get('phosphorous', '').strip()
        potassium   = request.form.get('potassium', '').strip()

        # ---- VALIDATION ----
        # Check for empty fields
        if not all([soil_type, crop_type, temperature, humidity,
                    moisture, nitrogen, phosphorous, potassium]):
            raise ValueError("All fields are required. Please fill in every input.")

        # Check dropdowns are valid
        if soil_type not in SOIL_TYPES:
            raise ValueError(f"Invalid soil type: '{soil_type}'.")
        if crop_type not in CROP_TYPES:
            raise ValueError(f"Invalid crop type: '{crop_type}'.")

        # Convert numeric inputs
        temperature = float(temperature)
        humidity    = float(humidity)
        moisture    = float(moisture)
        nitrogen    = float(nitrogen)
        phosphorous = float(phosphorous)
        potassium   = float(potassium)

        # Range validation
        if not (0 <= temperature <= 60):
            raise ValueError("Temperature must be between 0°C and 60°C.")
        if not (0 <= humidity <= 100):
            raise ValueError("Humidity must be between 0% and 100%.")
        if not (0 <= moisture <= 100):
            raise ValueError("Moisture must be between 0% and 100%.")
        if not (0 <= nitrogen <= 140):
            raise ValueError("Nitrogen must be between 0 and 140.")
        if not (0 <= phosphorous <= 145):
            raise ValueError("Phosphorous must be between 0 and 145.")
        if not (0 <= potassium <= 205):
            raise ValueError("Potassium must be between 0 and 205.")

        # ---- ENCODE CATEGORICAL INPUTS ----
        soil_enc = encoders['soil'].transform([soil_type])[0]
        crop_enc = encoders['crop'].transform([crop_type])[0]

        # ---- BUILD FEATURE ARRAY ----
        import pandas as pd_pred
        features = pd_pred.DataFrame([[
            temperature,
            humidity,
            moisture,
            soil_enc,
            crop_enc,
            nitrogen,
            phosphorous,
            potassium
        ]], columns=[
            'Temperature', 'Humidity', 'Moisture',
            'Soil_Type_enc', 'Crop_Type_enc',
            'Nitrogen', 'Phosphorous', 'Potassium'
        ])

        # ---- PREDICT ----
        prediction_enc = model.predict(features)[0]
        fertilizer_name = encoders['fertilizer'].inverse_transform([prediction_enc])[0]

        # Get prediction probabilities for confidence display
        probabilities = model.predict_proba(features)[0]
        confidence = round(max(probabilities) * 100, 2)

        # Get fertilizer additional info
        fert_info = FERTILIZER_INFO.get(fertilizer_name, {
            'description': 'A suitable fertilizer for your crop and soil conditions.',
            'usage': 'Follow standard agricultural guidelines for application.',
            'color': '#27ae60'
        })

        # Package user inputs for display on result page
        user_inputs = {
            'Soil Type'    : soil_type,
            'Crop Type'    : crop_type,
            'Temperature'  : f"{temperature} °C",
            'Humidity'     : f"{humidity} %",
            'Moisture'     : f"{moisture} %",
            'Nitrogen (N)' : nitrogen,
            'Phosphorous (P)': phosphorous,
            'Potassium (K)': potassium
        }

        return render_template('result.html',
                               fertilizer=fertilizer_name,
                               confidence=confidence,
                               fert_info=fert_info,
                               user_inputs=user_inputs)

    except ValueError as ve:
        # Show validation errors back on the prediction form
        return render_template('predict.html',
                               soil_types=SOIL_TYPES,
                               crop_types=CROP_TYPES,
                               error=str(ve))

    except Exception as e:
        # Catch any unexpected errors
        return render_template('predict.html',
                               soil_types=SOIL_TYPES,
                               crop_types=CROP_TYPES,
                               error=f"Unexpected error: {str(e)}. Please try again.")

# ----------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------
if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  Fertilizer Recommendation System")
    print("  Starting Flask Server...")
    print("=" * 55)
    if model is None:
        print("  ⚠️  WARNING: Model not found!")
        print("  Please run: python model/train_model.py")
    else:
        print("  ✅  Model loaded successfully.")
    print("  🌐  Open your browser at: http://127.0.0.1:8000")
    print("=" * 55 + "\n")
    app.run(debug=True, port=8000)
