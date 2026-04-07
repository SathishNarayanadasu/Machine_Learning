import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("../data/raw/AgriGo_dataset.csv")

# 🔍 Check columns
print("Columns in dataset:", df.columns)

# Add weight noise
df["Weight"] = df["Weight"] + np.random.uniform(-30, 30, size=len(df))

# Encoders
crop_encoder = LabelEncoder()
vehicle_encoder = LabelEncoder()

# ✅ USE CORRECT COLUMN NAMES
df["CropType"] = crop_encoder.fit_transform(df["CropType"])
df["VehicleType"] = vehicle_encoder.fit_transform(df["VehicleType"])

# Add 5% label noise
noise_fraction = 0.08
n_samples = int(len(df) * noise_fraction)
indices = random.sample(range(len(df)), n_samples)

for i in indices:
    df.loc[i, "VehicleType"] = random.choice(df["VehicleType"].unique())

# Split
X = df[["CropType", "Weight"]]
y = df["VehicleType"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save encoders
joblib.dump(crop_encoder, "../models/crop_encoder.pkl")
joblib.dump(vehicle_encoder, "../models/vehicle_encoder.pkl")

print("Preprocessing completed successfully.")