# 🚜 AgriGo – Crop Transportation Vehicle Recommendation System

## 📌 Project Overview

AgriGo is a Machine Learning–based system designed to recommend the most suitable transportation vehicle for agricultural crops based on:

* 🌾 Crop Type
* ⚖ Weight of the Crop

The system uses a **Decision Tree Classifier** trained on structured agricultural transportation data.
The trained model is deployed using a **Flask REST API**, enabling integration with an Android application.

---

# 🧠 What is Machine Learning?

Machine Learning (ML) is a subset of Artificial Intelligence (AI) that enables systems to learn patterns from data and make predictions without being explicitly programmed.

Instead of writing rules manually, we:

1. Provide historical data
2. Train a model
3. Allow the model to learn patterns
4. Use the model to predict new outcomes

---

# 📂 Project Structure

```
ML/
│
├── api/
│   └── app.py
    ├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── crop_transport_dataset.csv
│   └── processed/
│
├── models/
│   ├── crop_encoder.pkl
│   ├── vehicle_encoder.pkl
│   └── vehicle_prediction_model.pkl
│
├── training/
│   ├── eda.py
│   ├── preprocess.py
│   └── train_model.py
│
├── venv/
└── README.md
```

---

# ⚙ PHASE 1 — Environment Setup

## 🔹 What is a Virtual Environment?

A Virtual Environment is an isolated Python workspace that allows project-specific dependencies without affecting the system Python.

### Why is it important?

* Prevents version conflicts
* Ensures reproducibility
* Makes project portable

---

## 🔹 Step 1: Create Virtual Environment

```bash
python -m venv venv
```

## 🔹 Step 2: Activate Environment

```bash
venv\Scripts\activate
```

---

## 🔹 Step 3: Install Required Libraries

```bash
pip install pandas numpy scikit-learn flask joblib
```

## 🔹 Save Dependencies

```bash
pip freeze > requirements.txt
```

---

# 📦 Libraries Used & Purpose

## 1️⃣ pandas

Used for:

* Reading CSV files
* Data manipulation
* Data cleaning
* Data analysis

Definition:
Pandas is a Python library used for handling structured data in tabular form.

---

## 2️⃣ numpy

Used for:

* Numerical operations
* Mathematical computations
* Backend support for ML algorithms

Definition:
NumPy provides support for arrays and high-performance mathematical operations.

---

## 3️⃣ scikit-learn

Used for:

* Model training
* Decision Tree algorithm
* Train-test split
* Evaluation metrics

Definition:
Scikit-learn is an industry-standard Machine Learning library.

---

## 4️⃣ joblib

Used for:

* Saving trained models
* Loading models for deployment

Definition:
Joblib is used for serializing Python objects efficiently.

---

## 5️⃣ Flask

Used for:

* Creating REST API
* Connecting ML model to Android app
* Deployment backend

Definition:
Flask is a lightweight Python web framework.

---

# 📊 PHASE 2 — Data Understanding (EDA)

EDA stands for Exploratory Data Analysis.

## Purpose of EDA:

* Understand dataset structure
* Identify missing values
* Detect data inconsistencies
* Identify unique categories
* Validate dataset quality

---

## Key EDA Checks:

### 1. Shape of Dataset

Shows number of rows and columns.

### 2. Data Types

Ensures numeric and categorical data are correctly identified.

### 3. Missing Values

Important because ML models cannot handle null values.

### 4. Unique Values

Helps understand categorical classes.

---

# 🧹 PHASE 3 — Data Preprocessing

Data preprocessing prepares raw data for machine learning.

---

## Why Preprocessing is Necessary?

Machine learning models:

* Cannot handle text directly
* Require numerical inputs
* Perform better with clean data

---

## Steps in Preprocessing

### 1️⃣ Label Encoding

Definition:
Label Encoding converts categorical text values into numerical form.

Example:

```
Apple → 0
Ginger → 1
```

Why?
Because ML models only understand numbers.

---

### 2️⃣ Feature Selection

Features used:

* Crop
* Weight

Target:

* Vehicle

---

### 3️⃣ Train-Test Split

Definition:
Splitting data into training and testing sets.

```
80% → Training
20% → Testing
```

Purpose:

* Train on one portion
* Test on unseen data
* Measure model performance

---

# 🌳 PHASE 4 — Model Training

Model Used:
Decision Tree Classifier

---

## What is a Decision Tree?

A Decision Tree is a supervised machine learning algorithm that:

* Splits data based on feature values
* Creates rule-based structure
* Makes predictions by following branches

Example rule:

```
IF Crop = Apple AND Weight > 1000 → Truck
```

---

## Why Decision Tree?

* Easy to interpret
* Works well for structured data
* Handles both categorical & numerical data
* Good for rule-based prediction systems

---

## Model Training Steps

1. Load dataset
2. Encode categorical values
3. Split into train-test
4. Initialize Decision Tree
5. Train model
6. Evaluate accuracy
7. Save model using joblib

---

## Evaluation Metrics

### 1️⃣ Training Accuracy

How well model performs on training data.

### 2️⃣ Testing Accuracy

How well model performs on unseen data.

### 3️⃣ Classification Report

Includes:

* Precision
* Recall
* F1-score

---

# 💾 PHASE 5 — Model Saving

We save:

* crop_encoder.pkl
* vehicle_encoder.pkl
* vehicle_prediction_model.pkl

Why?

Because training should not happen every time server runs.

Deployment loads saved model instead.

---

# 🌐 PHASE 6 — Deployment using Flask API

Deployment means making the model accessible via web.

---

## API Endpoints

### GET /crops

Returns list of available crops.

Purpose:
Android dropdown population.

---

### POST /predict

Input:

```json
{
  "crop": "Apple",
  "weight": 1200
}
```

Output:

```json
{
  "recommended_vehicle": "Truck"
}
```

---

# 🔄 Complete System Flow

1. User selects crop & weight in Android
2. Android sends request to Flask API
3. Flask receives data
4. Crop is encoded
5. Model predicts vehicle
6. Vehicle is decoded
7. JSON response returned

---

# 🚀 Why This System is Useful

* Reduces manual decision making
* Optimizes transportation
* Saves cost
* Improves logistics planning
* Demonstrates real-world ML deployment

---

# 📈 Future Improvements

* Add more features (distance, road type)
* Use Random Forest for better accuracy
* Deploy on cloud (AWS/Render)
* Add database storage
* Add user authentication

---
"What is happening in  project?"

Answer:

> The system uses a Decision Tree classifier trained on crop and weight data to predict the most suitable transportation vehicle. The model is deployed using a Flask REST API, which allows real-time integration with an Android application.

---

# 📌 Conclusion

This project demonstrates:

* End-to-end ML pipeline
* Data preprocessing
* Model training & evaluation
* Model persistence
* REST API deployment
* Mobile integration readiness

It follows standard machine learning workflow from raw data to deployed system.

