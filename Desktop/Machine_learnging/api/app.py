from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# ------------------------------
# Base Directory
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------
# Model Paths
# ------------------------------
model_path = os.path.join(BASE_DIR, "models", "vehicle_prediction_model.pkl")
crop_encoder_path = os.path.join(BASE_DIR, "models", "crop_encoder.pkl")
vehicle_encoder_path = os.path.join(BASE_DIR, "models", "vehicle_encoder.pkl")

# ------------------------------
# Load Model & Encoders
# ------------------------------
model = joblib.load(model_path)
crop_encoder = joblib.load(crop_encoder_path)
vehicle_encoder = joblib.load(vehicle_encoder_path)

print("🚀 Server Ready...")

# ------------------------------
# HOME ROUTE (IMPORTANT)
# ------------------------------
@app.route("/")
def home():
    return "API WORKING"

# ------------------------------
# TEST ROUTE (OPTIONAL DEBUG)
# ------------------------------
@app.route("/test")
def test():
    return "TEST OK"

# ------------------------------
# PREDICTION ROUTE
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"})

        if "crop" not in data or "weight" not in data:
            return jsonify({"error": "Both 'crop' and 'weight' are required"})

        crop = data["crop"].strip().title()
        weight = float(data["weight"])

        # Validate weight
        if weight <= 0:
            return jsonify({"error": "Weight must be greater than 0"})

        # Validate crop
        if crop not in crop_encoder.classes_:
            return jsonify({
                "error": f"Invalid crop. Allowed crops: {list(crop_encoder.classes_)}"
            })

        # Encode crop
        crop_encoded = crop_encoder.transform([crop])[0]

        # Predict
        prediction = model.predict([[crop_encoded, weight]])

        # Decode vehicle
        vehicle = vehicle_encoder.inverse_transform(prediction)[0]

        # FINAL RESPONSE (IMPORTANT KEY)
        return jsonify({
            "vehicle": vehicle
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)