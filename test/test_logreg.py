"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
# (you will probably need to import more things here)
import numpy as np
import sys
import os

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from regression.logreg import LogisticRegressor
from regression.utils import loadDataset

#from main
features = [
    'Penicillin V Potassium 500 MG',
    'Computed tomography of chest and abdomen',
    'Plain chest X-ray (procedure)',
    'Low Density Lipoprotein Cholesterol',
    'Creatinine',
    'AGE_DIAGNOSIS'
]

# data in csv with 80/20 divide
X_train, X_val, y_train, y_val = loadDataset(
    features=features,
    split_percent=0.8,
    split_seed=7
)

# total num of geatures
num_feats = X_train.shape[1]

def test_prediction():
    #make model 
    model = LogisticRegressor(num_feats=num_feats)

    bias_column = np.ones((X_val.shape[0], 1))
    X_padded = np.hstack([X_val, bias_column])

    predictions = model.make_prediction(X_padded)

    assert len(predictions) == len(X_val), \
        "One prediction per sample"

    for p in predictions:
        assert p >= 0 and p <= 1, \
            "Prediction between 1 and 0"

    # make input matrix high so predictions are extreme
    model.W = np.ones(num_feats + 1) * 38859834028

    # positive weights = prediciton of 1
    X_positive = np.ones((1, num_feats + 1))
    assert model.make_prediction(X_positive)[0] > 0.99, \
        "Large positive input should give a prediction close to 1"

    # negative = 0
    X_negative = -np.ones((1, num_feats + 1))
    assert model.make_prediction(X_negative)[0] < 0.01, \
        "Large negative input should give a prediction close to 0"


def test_loss_function():
    model = LogisticRegressor(num_feats=num_feats)

    y_true = np.array([1.0, 0.0])

    # Good predictions: model is nearly correct
    y_pred_good = np.array([0.999, 0.001])

    # Bad predictions: model is completely wrong
    y_pred_bad = np.array([0.001, 0.999])

    loss_good = model.loss_function(y_true, y_pred_good)
    loss_bad  = model.loss_function(y_true, y_pred_bad)

    assert loss_good < loss_bad, \
        "In this example true"

def test_gradient():
    
    model = LogisticRegressor(num_feats=num_feats)

    # bias column
    bias_column = np.ones((X_train.shape[0], 1))
    X_padded = np.hstack([X_train, bias_column])

    gradient = model.calculate_gradient(y_train, X_padded)

    # loss before
    loss_before = model.loss_function(y_train, model.make_prediction(X_padded))

    # take a step 
    model.W = model.W - 0.000001 * gradient

    # loss after 
    loss_after = model.loss_function(y_train, model.make_prediction(X_padded))

    # Loss should go down after one step
    assert loss_after < loss_before, \
        "Taking a gradient descent step should reduce the training loss"

def test_training():

    model = LogisticRegressor(
        num_feats=num_feats,
        learning_rate=0.01,
        tol=0.001,
        max_iter=100,
        batch_size=10
    )

    # raw weights 
    weights_before_training = model.W.copy()

    # training on csv
    model.train_model(X_train, y_train, X_val, y_val)

    # weights changing
    weights_changed = False
    for i in range(len(model.W)):
        if model.W[i] != weights_before_training[i]:
            weights_changed = True
    assert weights_changed, "weights changed after training"

    # model improvement
    assert model.loss_hist_train[-1] < model.loss_hist_train[0], \
        "Training improved model with less loss"