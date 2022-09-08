import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix 

def evaluate(model, X_test, y_test, labels, history):
    """
    Evaluate a Keras model and plot the results

    Args:
        model: a Keras model
        X_test, y_test: the test data
        label: the actual labels
        history: the output of model.fit
    """

    print(pd.Series(model.evaluate(X_test, y_test), index=model.metrics_names))
        
    fig = plt.figure(figsize=(8, 6))
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    fig.savefig("history.png")
    
    probabilities = model.predict(X_test)
    predicted_classes = np.argmax(probabilities, axis=1)
    if y_test.shape[1] == len(labels):
        y_test = np.argmax(y_test, axis=1)

    confMat = pd.DataFrame(confusion_matrix(y_test, predicted_classes), index=labels, columns=labels)
    confMat /= confMat.sum(axis=1)

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(confMat, cmap=plt.cm.Blues, annot=True, cbar=False)
    plt.xlabel("Predicted labels", fontsize=14)
    plt.ylabel("True labels", fontsize=14)
    plt.title("Confusion matrix", fontsize=18)
    fig.savefig("confusion.png")
