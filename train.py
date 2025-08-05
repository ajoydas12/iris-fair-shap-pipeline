import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load data
data = pd.read_csv("data/iris.csv")
X = data[['sepal_length','sepal_width','petal_length','petal_width']]
y = data['species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train model
model = DecisionTreeClassifier(max_depth=3, random_state=1)
model.fit(X_train, y_train)

# Save model and test data
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/decision_tree_model.joblib")
joblib.dump((X_test, y_test), "models/test_data.joblib")

print("Model and test data saved.")

