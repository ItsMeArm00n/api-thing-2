import os
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# Google Drive direct download link
MODEL_FILE = "Advanced_air_pollution_model.pkl"
DRIVE_FILE_ID = "1mHUgSz9OREZcUIW2isUk3AMQpmGz2agx"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

# Download the model if it doesn't exist
if not os.path.exists(MODEL_FILE):
    print("📥 Downloading model from Google Drive...")
    response = requests.get(MODEL_URL, stream=True)
    if response.status_code == 200:
        with open(MODEL_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Model downloaded successfully!")
    else:
        raise RuntimeError(f"❌ Failed to download model, status code {response.status_code}")

# Load model
try:
    model = joblib.load(MODEL_FILE)
    print("✅ Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Error loading model: {e}")


@app.route('/predict', methods=['POST'], strict_slashes=False)
def predict():
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
