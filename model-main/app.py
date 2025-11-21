from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "🚀 SafeRoute AI Model is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    # Receive input JSON data
    data = request.get_json()

    # Convert data to NumPy array and reshape
    features = np.array(data["features"]).reshape(1, -1)

    # Scale input data
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)

    # Return result
    return jsonify({"prediction": int(prediction[0])})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
