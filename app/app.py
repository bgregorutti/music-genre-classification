from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

MODEL = None
LABELS = np.array(["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])

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
    # app.logger.info("Classify service")
    # app.logger.info(f"Receiving data of shape: {data.shape}")
    if data.shape[0] > 1:
        app.logger.warning("Trying to predict on two or more data. Not implemented yet...")

    if not data.size:
        app.logger.warning("Empty data")
        return jsonify({})

    predictions = MODEL.predict(data, verbose=0)
    predicted_label = LABELS[np.argmax(predictions)]
    return jsonify({"predicted_label": predicted_label, "probability": float(np.max(predictions))})

@app.route("/classify_overall", methods=["POST"])
def classify_overall(): #TODO push a table of the guessed genres with probabilities
    data = np.array(request.json)
    app.logger.info("Classify service")
    app.logger.info(f"Receiving data of shape: {data.shape}")
    data = data[:, :130, :, :] #TODO to be fixed

    # Predict each sequence and get the corresponding labels
    predictions = MODEL.predict(data, verbose=0)
    predicted_labels = LABELS[np.argmax(predictions, axis=1)]

    # Compute the number of times each label is predicted
    predicted_labels, probabilities = np.unique(predicted_labels, return_counts=True)
    probabilities = probabilities / np.sum(probabilities)

    return jsonify({"predicted_labels": list(predicted_labels), "probabilities": list(probabilities)})

# def most_frequent(arr):
#     values, counts = np.unique(arr, return_counts=True)
#     return values[np.argmax(counts)]

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
