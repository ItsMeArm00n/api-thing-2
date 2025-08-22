import os
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load model
model = None
try:
    model = joblib.load(r"C:\Users\armaa\Documents\GitHub\api-thing-2\Advanced_air_pollution_model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route('/predict', methods=['POST'], strict_slashes=False)
def predict():
  #  if not model:
   #     return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        required_fields = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required field(s)"}), 400

        input_features = [[
            float(data["PM2.5"]),
            float(data["PM10"]),
            float(data["NO2"]),
            float(data["SO2"]),
            float(data["CO"]),
            float(data["O3"])
        ]]

        predicted_aqi = model.predict(input_features)[0]
        return jsonify({"aqi": predicted_aqi})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
