from flask import Flask, request, jsonify
import joblib
import os
app = Flask(__name__)
# ------------------------------
# Get Base Directory
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ------------------------------
# Model Paths
# ------------------------------
model_path = os.path.join(BASE_DIR, "models", "vehicle_prediction_model.pkl")
crop_encoder_path = os.path.join(BASE_DIR, "models", "crop_encoder.pkl")
vehicle_encoder_path = os.path.join(BASE_DIR, "models", "vehicle_encoder.pkl")
# ------------------------------
# Load Model and Encoders
# ------------------------------
model = joblib.load(model_path)
crop_encoder = joblib.load(crop_encoder_path)
vehicle_encoder = joblib.load(vehicle_encoder_path)
# PRINT AVAILABLE CROPS (DEBUGGING)
print("\nAvailable crops in encoder:")
for c in crop_encoder.classes_:
    print(f"'{c}'")
print("\nServer Ready...\n")
# ------------------------------
# Prediction Route
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        crop = data["crop"].strip()
        weight = float(data["weight"])
        # Check if crop exists       
        if crop not in crop_encoder.classes_:
            return jsonify({
                "error": f"Invalid crop. Allowed crops: {list(crop_encoder.classes_)}"
            })
        crop_encoded = crop_encoder.transform([crop])[0]
        prediction = model.predict([[crop_encoded, weight]])
        vehicle = vehicle_encoder.inverse_transform(prediction)[0]
        return jsonify({
            "recommended_vehicle": vehicle
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        })
#-------------------------------
#Run App
#-------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)