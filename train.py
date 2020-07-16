import json
import os

from joblib import dump
import matplotlib as plt
import numpy as np
from sklearn import ensemble, datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

#load directories for model and metadata
MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_FILE = os.environ["METADATA_FILE"]
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)

#load and split data
print("loading...")
boston = datasets.load_boston()

print("splitting data...")
X, y = shuffle(boston.data, boston.target, random_state = 18)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

#fitting the regression
print("fitting model...")
params = {
    'n_estimators':500,
    'max_depth':3,
    'min_samples_split':2,
    'learning_rate': 0.01,
    'loss':'ls'
}

clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
train_mse = mean_squared_error(y_train, clf.predict(X_train))
test_mse = mean_squared_error(y_test, clf.predict(X_test))
metadata = {
    "train_mse" : train_mse,
    "test_mse" : test_mse
}

#serializing
print("Serializing model to: {}".format(MODEL_PATH))
dump(clf, MODEL_PATH)

print("Serializing metadata to: {}".format(METADATA_PATH))
with open(METADATA_PATH, 'w') as out_file:
    json.dump(metadata, out_file)