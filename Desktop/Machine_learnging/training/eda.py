import pandas as pd

# Load raw dataset
df = pd.read_csv("../data/raw/crop_transport_dataset.csv")

print("🔹 First 5 Rows:")
print(df.head())

print("\n🔹 Dataset Shape:")
print(df.shape)

print("\n🔹 Data Types:")
print(df.dtypes)

print("\n🔹 Missing Values:")
print(df.isnull().sum())

print("\n🔹 Unique Crops:")
print(df["Crop"].unique())

print("\n🔹 Unique Vehicles:")
print(df["Vehicle"].unique())