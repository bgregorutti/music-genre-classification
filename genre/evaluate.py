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
        
    plt.figure(figsize=(12,8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    
    probabilities = model.predict(X_test)
    predicted_classes = np.argmax(probabilities, axis=1)
    if y_test.shape[1] == len(labels):
        y_test = np.argmax(y_test, axis=1)

    confMat = pd.DataFrame(confusion_matrix(y_test, predicted_classes), index=labels, columns=labels)
    confMat /= np.sum(confMat, axis=1)

    plt.figure(figsize=(12,8))
    sns.heatmap(confMat, cmap=plt.cm.Blues, annot=True)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title('Confusion matrix')

    plt.show()
