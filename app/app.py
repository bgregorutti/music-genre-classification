from flask import Flask, request
from keras.models import load_model

app = Flask(__name__)
MODEL = None

@app.before_first_request
def before_first_request():
    global MODEL
    app.logger.warning("Loading Keras model...")
    MODEL = load_model("../model")

@app.route('/')
def index():
    return "This is an example app"

@app.route('/classify', methods=["POST"])
def classify():
    data = request.json
    print(data)
    return "Classifying"

if __name__ == "__main__":
    app.run(debug=True)
