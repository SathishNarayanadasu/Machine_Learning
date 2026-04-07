import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

# -----------------------------
# Load Dataset (NO EXTRA SPACE!)
# -----------------------------
df = pd.read_csv("../data/raw/AgriGo_dataset.csv")

# -----------------------------
# Add Moderate Weight Noise
# -----------------------------
df["Weight"] = df["Weight"] + np.random.uniform(-30, 30, size=len(df))

# -----------------------------
# Load Encoders
# -----------------------------
crop_encoder = joblib.load("../models/crop_encoder.pkl")
vehicle_encoder = joblib.load("../models/vehicle_encoder.pkl")

# -----------------------------
# Encode Columns (Use Correct Names)
# -----------------------------
df["CropType"] = crop_encoder.transform(df["CropType"])
df["VehicleType"] = vehicle_encoder.transform(df["VehicleType"])

# -----------------------------
# Split Data
# -----------------------------
X = df[["CropType", "Weight"]]
y = df["VehicleType"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Controlled Decision Tree
# -----------------------------
model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=30,
    min_samples_leaf=15,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))

y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "../models/vehicle_prediction_model.pkl")

print("\nModel saved successfully.")