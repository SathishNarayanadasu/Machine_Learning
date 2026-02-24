import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load raw dataset
df = pd.read_csv("../data/raw/crop_transport_dataset.csv")

# -----------------------------
# 🔥 Add Moderate Real-World Noise
# -----------------------------
df["Weight"] = df["Weight"] + np.random.uniform(-20, 20, size=len(df))

# Initialize encoders
crop_encoder = LabelEncoder()
vehicle_encoder = LabelEncoder()

# Encode Crop feature
df["Crop"] = crop_encoder.fit_transform(df["Crop"])

# Encode target variable
df["Vehicle"] = vehicle_encoder.fit_transform(df["Vehicle"])

# Separate features and target
X = df[["Crop", "Weight"]]
y = df["Vehicle"]

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save encoders
joblib.dump(crop_encoder, "../models/crop_encoder.pkl")
joblib.dump(vehicle_encoder, "../models/vehicle_encoder.pkl")

print("Preprocessing completed successfully.")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])