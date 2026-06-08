# 🌿 Fertilizer Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-orange.svg)](https://scikit-learn.org)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple.svg)](https://getbootstrap.com)

A **Machine Learning powered web application** that recommends the most suitable fertilizer
for your crop based on soil nutrients and environmental conditions — using **Logistic Regression**.

> **B.Tech 3rd Year Project | CSM355 Machine Learning | Lovely Professional University (LPU)**

---

## 📌 Project Overview

Farmers often struggle to determine the right fertilizer for their crops. Incorrect fertilizer
use leads to reduced yield, wasted money, and soil damage. This system solves that problem
by predicting the best fertilizer using a trained Logistic Regression ML model.

**Input → ML Model → Fertilizer Recommendation**

---

## ✨ Features

- 🤖 Logistic Regression-based fertilizer prediction
- 🌾 Supports 7 fertilizer types (Urea, DAP, 14-35-14, 17-17-17, 20-20, 28-28, Super Phosphate)
- 🏔️ 5 soil types and 6 crop types supported
- 📊 Displays confidence score for each prediction
- 📱 Fully responsive design with Bootstrap 5
- ✅ Input validation (client-side + server-side)
- 🎨 Professional, modern UI

---

## 🗂️ Project Structure

```
Fertilizer-Recommendation-System/
│
├── app.py                        ← Flask web application (main entry point)
│
├── model/
│   ├── train_model.py            ← ML training script
│   ├── fertilizer_model.pkl      ← Trained Logistic Regression model
│   └── encoder.pkl               ← Label encoders (soil, crop, fertilizer)
│
├── dataset/
│   └── fertilizer.csv            ← Training dataset (130 records)
│
├── templates/
│   ├── base.html                 ← Base layout (navbar + footer)
│   ├── index.html                ← Home page
│   ├── about.html                ← About project page
│   ├── predict.html              ← Prediction form page
│   └── result.html               ← Prediction result page
│
├── static/
│   ├── css/
│   │   └── style.css             ← Custom stylesheet
│   ├── js/                       ← JavaScript (optional)
│   └── images/                   ← Images folder
│
├── requirements.txt              ← Python dependencies
└── README.md                     ← This file
```

---

## 🚀 Installation & Setup

### Step 1 — Clone or Download the Project

```bash
git clone https://github.com/yourusername/Fertilizer-Recommendation-System.git
cd Fertilizer-Recommendation-System
```

### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Train the Machine Learning Model

```bash
python model/train_model.py
```

This creates two files in the `model/` folder:
- `fertilizer_model.pkl` — trained Logistic Regression model
- `encoder.pkl` — label encoders for soil, crop and fertilizer

### Step 5 — Run the Flask Application

```bash
python app.py
```

### Step 6 — Open in Browser

```
http://127.0.0.1:5000
```

---

## 🧪 Input Features

| Feature | Type | Description |
|---------|------|-------------|
| Temperature | Numeric | Ambient temperature in °C |
| Humidity | Numeric | Relative humidity in % |
| Moisture | Numeric | Soil moisture in % |
| Soil Type | Categorical | Black / Clayey / Loamy / Red / Sandy |
| Crop Type | Categorical | Cotton / Maize / Paddy / Sugarcane / Tobacco / Wheat |
| Nitrogen (N) | Numeric | Soil nitrogen level (kg/ha) |
| Phosphorous (P) | Numeric | Soil phosphorous level (kg/ha) |
| Potassium (K) | Numeric | Soil potassium level (kg/ha) |

---

## 🎯 Sample Test Inputs

| Soil | Crop | Temp | Humidity | Moisture | N | P | K | Result |
|------|------|------|----------|----------|---|---|---|--------|
| Sandy | Maize | 26 | 52 | 38 | 37 | 0 | 0 | **Urea** |
| Loamy | Sugarcane | 29 | 52 | 45 | 12 | 0 | 36 | **DAP** |
| Black | Cotton | 34 | 65 | 62 | 7 | 9 | 9 | **14-35-14** |
| Red | Wheat | 27 | 55 | 45 | 20 | 20 | 0 | **20-20** |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | 100% |
| Test Accuracy | 100% |
| 5-Fold Cross Validation | 100% |
| F1 Score | 1.00 |

---

## 🛠️ Technologies Used

- **Python 3.x** — Core programming language
- **Flask** — Web framework for backend
- **Scikit-learn** — Machine Learning library
- **Pandas** — Data manipulation
- **NumPy** — Numerical computing
- **Pickle** — Model serialization
- **Bootstrap 5** — Frontend CSS framework
- **HTML5 / CSS3** — Web markup and styling

---

## 📸 Screenshots

| Page | Description |
|------|-------------|
| Home | Hero section, features, how-it-works, stats bar |
| About | Problem statement, dataset info, algorithm explanation |
| Predict | Input form with soil, crop, and NPK fields |
| Result | Recommended fertilizer with confidence score and NPK bars |

---

## 👨‍💻 Author

**B.Tech CSE Student**
Lovely Professional University (LPU)
Subject: CSM355 — Machine Learning Project

---

## 📄 License

This project is created for academic purposes at Lovely Professional University.
For educational use only.
