from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

MODEL = None
LABELS = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

@app.before_first_request
def before_first_request():
    global MODEL
    app.logger.warning("Loading Keras model...")
    MODEL = load_model("../model")

@app.route("/")
def index():
    app.logger.warning("Index")
    return "Application running"

@app.route("/classify", methods=["POST"])
def classify():
    data = np.array(request.json)
    app.logger.info("Classifiy service")
    app.logger.info(f"Receiving data of shape: {data.shape}")
    if data.shape[0] > 1:
        app.logger.warning("Trying to predict on two or more data. Not implemented yet...")

    predictions = MODEL.predict(data)
    predicted_label = LABELS[np.argmax(predictions)]
    return jsonify({"predicted_label": predicted_label, "probability": float(np.max(predictions))})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
