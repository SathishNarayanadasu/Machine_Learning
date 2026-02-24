import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("../data/raw/crop_transport_dataset.csv")

# 🔥 Add SAME moderate noise
df["Weight"] = df["Weight"] + np.random.uniform(-20, 20, size=len(df))

# Load encoders
crop_encoder = joblib.load("../models/crop_encoder.pkl")
vehicle_encoder = joblib.load("../models/vehicle_encoder.pkl")

# Encode again (for consistency)
df["Crop"] = crop_encoder.transform(df["Crop"])
df["Vehicle"] = vehicle_encoder.transform(df["Vehicle"])

# Features & Target
X = df[["Crop", "Weight"]]
y = df["Vehicle"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 Balanced Tree Depth
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "../models/vehicle_prediction_model.pkl")

print("\nModel saved successfully.")