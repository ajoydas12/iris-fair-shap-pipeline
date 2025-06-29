import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# test_model_accuracy()
def evaluate_model(pkl_path, X, y):
    try:
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        print(f"{pkl_path}: accuracy = {acc:.3f}")
        print(f"Confusion Matrix:\n{cm}\n")
        return acc, cm
    except Exception as e:
        print(f"{pkl_path}: ERROR - {e}")
        return None

def test_all_model_versions():
    df = pd.read_csv('data/iris.csv')
    X = df[['sepal_length','sepal_width','petal_length','petal_width']]
    y = df['species']
    accs = {}
    for version in ["v1", "v2", "v3"]:
        pkl_path = f"models/decision_tree_model_{version}.pkl"
        acc, cm = evaluate_model(pkl_path, X, y)
        accs[version] = acc
    # Check at least one model meets the threshold
    assert any(acc and acc > 0.7 for acc in accs.values()), f"No model version has accuracy > 0.7: {accs}"

if __name__ == "__main__":
    test_all_model_versions()
    print("Model accuracy comparison completed.")