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

    predictions = MODEL.predict(data, verbose=0)
    predicted_label = LABELS[np.argmax(predictions)]
    return jsonify({"predicted_label": predicted_label, "probability": float(np.max(predictions))})

@app.route("/classify_overall", methods=["POST"])
def classify_overall(): #TODO push a table of the guessed genres with probabilities
    data = np.array(request.json)
    app.logger.info("Classify service")
    app.logger.info(f"Receiving data of shape: {data.shape}")
    data = data[:, :130, :, :] #TODO to be fixed

    predictions = MODEL.predict(data, verbose=0)

    # predicted_label
    predicted_labels = LABELS[np.argmax(predictions, axis=1)]
    predicted_label = most_frequent(predicted_labels)

    # probability
    probability = np.mean(np.max(predictions[predicted_labels == predicted_label], axis=1))

    app.logger.info(predicted_label)
    app.logger.info(np.max(predictions[predicted_labels == predicted_label], axis=1))

    return jsonify({"predicted_label": predicted_label, "probability": float(probability)})

def most_frequent(arr):
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
