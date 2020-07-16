import os
from joblib import load
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
METADATA_FILE = os.environ["METADATA_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)

def get_data():
    """
    Returns data for inference.
    """
    print("Loading data...")
    boston = datasets.load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=18)
    X = X.astype(np.float32)
    offset = int(X.shape[0] * 0.9)
    X_test, y_test = X[offset:], y[offset:]
    return X_test, y_test

print("Running inference")
X, y = get_data()

#Load model
print("loading model from: {}".format(MODEL_PATH))
clf = load(MODEL_PATH)

#run inference
print("Scoring...")
y_pred = clf.predict(X)
print(y_pred)