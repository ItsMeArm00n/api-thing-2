import requests
MODEL_FILE = "Advanced_air_pollution_model.pkl"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1mHUgSz9OREZcUIW2isUk3AMQpmGz2agx"

import os
if not os.path.exists(MODEL_FILE):
    print("📥 Downloading model...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_FILE, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✅ Model downloaded!")
