import os
import joblib
import pytest
from sklearn.metrics import accuracy_score, precision_score, f1_score

@pytest.fixture(scope="module")
def load_model_and_data():
    model_path = "models/model.joblib"
    test_data_path = "models/test_data.joblib"

    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    assert os.path.exists(test_data_path), f"Test data file not found: {test_data_path}"

    model = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)

    return model, X_test, y_test

def test_model_accuracy(load_model_and_data):
    model, X_test, y_test = load_model_and_data
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    assert acc >= 0.9, f"Accuracy below threshold: {acc:.3f}"

def test_model_precision(load_model_and_data):
    model, X_test, y_test = load_model_and_data
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average='macro')

    assert precision >= 0.9, f"Precision below threshold: {precision:.3f}"

def test_model_f1_score(load_model_and_data):
    model, X_test, y_test = load_model_and_data
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')

    assert f1 >= 0.9, f"F1 Score below threshold: {f1:.3f}"
