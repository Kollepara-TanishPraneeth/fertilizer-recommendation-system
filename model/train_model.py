# =============================================================
# train_model.py
# Fertilizer Recommendation System - Model Training Script
# B.Tech 3rd Year ML Project | LPU | CSM355
# =============================================================

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

print("=" * 60)
print("  FERTILIZER RECOMMENDATION SYSTEM - MODEL TRAINING")
print("=" * 60)

# ----------------------------------------------------------
# STEP 1: LOAD THE DATASET
# ----------------------------------------------------------
print("\n[STEP 1] Loading dataset...")

dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'fertilizer.csv')
df = pd.read_csv(dataset_path)

print(f"  Dataset loaded successfully!")
print(f"  Total rows    : {df.shape[0]}")
print(f"  Total columns : {df.shape[1]}")
print(f"\n  First 5 rows:")
print(df.head())

# ----------------------------------------------------------
# STEP 2: DATA CLEANING - CHECK FOR MISSING VALUES
# ----------------------------------------------------------
print("\n[STEP 2] Checking for missing values...")

missing = df.isnull().sum()
if missing.sum() == 0:
    print("  No missing values found. Dataset is clean!")
else:
    print("  Missing values detected:")
    print(missing[missing > 0])
    # Drop rows with missing values
    df.dropna(inplace=True)
    print(f"  Rows after dropping nulls: {df.shape[0]}")

# ----------------------------------------------------------
# STEP 3: LABEL ENCODING (Convert text columns to numbers)
# ----------------------------------------------------------
print("\n[STEP 3] Applying Label Encoding to categorical columns...")

# We need to encode: Soil_Type, Crop_Type, Fertilizer_Name
le_soil   = LabelEncoder()
le_crop   = LabelEncoder()
le_fert   = LabelEncoder()

df['Soil_Type_enc']  = le_soil.fit_transform(df['Soil_Type'])
df['Crop_Type_enc']  = le_crop.fit_transform(df['Crop_Type'])
df['Fertilizer_enc'] = le_fert.fit_transform(df['Fertilizer_Name'])

print(f"  Soil Types   : {list(le_soil.classes_)}")
print(f"  Crop Types   : {list(le_crop.classes_)}")
print(f"  Fertilizers  : {list(le_fert.classes_)}")

# ----------------------------------------------------------
# STEP 4: FEATURE SELECTION
# ----------------------------------------------------------
print("\n[STEP 4] Selecting features and target variable...")

# Features (X) - the input columns used for prediction
X = df[[
    'Temperature',
    'Humidity',
    'Moisture',
    'Soil_Type_enc',
    'Crop_Type_enc',
    'Nitrogen',
    'Phosphorous',
    'Potassium'
]]

# Target (y) - what we want to predict
y = df['Fertilizer_enc']

print(f"  Features shape : {X.shape}")
print(f"  Target shape   : {y.shape}")
print(f"  Feature columns: {list(X.columns)}")

# ----------------------------------------------------------
# STEP 5: TRAIN-TEST SPLIT (80% train, 20% test)
# ----------------------------------------------------------
print("\n[STEP 5] Splitting data into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing samples  : {X_test.shape[0]}")

# ----------------------------------------------------------
# STEP 6: TRAIN LOGISTIC REGRESSION MODEL
# ----------------------------------------------------------
print("\n[STEP 6] Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,       # Maximum number of iterations for convergence
    solver='lbfgs',      # Efficient solver for multi-class problems
    random_state=42
)

model.fit(X_train, y_train)
print("  Model trained successfully!")

# ----------------------------------------------------------
# STEP 7: MODEL EVALUATION
# ----------------------------------------------------------
print("\n[STEP 7] Evaluating model performance...")

# Predictions on test set
y_pred = model.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\n  ✅ Accuracy Score : {accuracy * 100:.2f}%")

# Confusion Matrix
print("\n  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classification Report
print("\n  Classification Report:")
target_names = list(le_fert.classes_)
report = classification_report(y_test, y_pred, target_names=target_names)
print(report)

# ----------------------------------------------------------
# STEP 8: CROSS VALIDATION
# ----------------------------------------------------------
print("\n[STEP 8] Performing 5-Fold Cross Validation...")

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"  CV Scores (each fold) : {[f'{s*100:.2f}%' for s in cv_scores]}")
print(f"  Mean CV Accuracy      : {cv_scores.mean() * 100:.2f}%")
print(f"  Standard Deviation    : {cv_scores.std() * 100:.2f}%")

# ----------------------------------------------------------
# STEP 9: SAVE MODEL AND ENCODERS WITH PICKLE
# ----------------------------------------------------------
print("\n[STEP 9] Saving model and encoders...")

# Directory to save model files
model_dir = os.path.dirname(__file__)

# Save the trained Logistic Regression model
model_path = os.path.join(model_dir, 'fertilizer_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"  Model saved      : {model_path}")

# Save all three label encoders together in a dict
encoder_path = os.path.join(model_dir, 'encoder.pkl')
encoders = {
    'soil'       : le_soil,
    'crop'       : le_crop,
    'fertilizer' : le_fert
}
with open(encoder_path, 'wb') as f:
    pickle.dump(encoders, f)
print(f"  Encoders saved   : {encoder_path}")

# ----------------------------------------------------------
# FINAL SUMMARY
# ----------------------------------------------------------
print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"  Model        : Logistic Regression")
print(f"  Accuracy     : {accuracy * 100:.2f}%")
print(f"  Mean CV Acc  : {cv_scores.mean() * 100:.2f}%")
print(f"  Fertilizers  : {list(le_fert.classes_)}")
print(f"  Files saved  :")
print(f"    → fertilizer_model.pkl")
print(f"    → encoder.pkl")
print("=" * 60)
print("\n  ✅ You can now run app.py to start the web application!\n")
