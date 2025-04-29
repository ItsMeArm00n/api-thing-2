import os
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load model (with auto-download if missing)
model = None
if not os.path.exists("Advanced_air_pollution_model.pkl"):
    print("Downloading model file...")
    try:
        import requests
        url = "https://github.com/ItsMeArm00n/api-thing-2/blob/master/Advanced_air_pollution_model.pkl"
        r = requests.get(url)
        with open("Advanced_air_pollution_model.pkl", "wb") as f:
            f.write(r.content)
    except Exception as e:
        print(f"Download failed: {e}")

try:
    model = joblib.load("Advanced_air_pollution_model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        required_fields = ["no2", "co", "so2", "o3", "pm25", "pm10", "temp", "humidity"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required field(s)"}), 400

        input_features = [[
            float(data["no2"]),
            float(data["co"]),
            float(data["so2"]),
            float(data["o3"]),
            float(data["pm25"]),
            float(data["pm10"]),
            float(data["temp"]),
            float(data["humidity"])
        ]]

        predicted_aqi = model.predict(input_features)[0]
        return jsonify({"aqi": predicted_aqi})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
