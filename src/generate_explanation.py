import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

shap.initjs()

# Load the model and data
model = joblib.load("artifacts/model.joblib")
df = pd.read_csv("data/iris.csv")
        
# Prepare data dynamically based on model's expected features
expected_features = model.feature_names_in_

print(f"Expected features: {expected_features}")
X_train, X_test, y_train, y_test = train_test_split(
    df[expected_features],
    df['species'],
    test_size=0.4,
    random_state=42,
    stratify=df['species']
)
# Initialize SHAP explainer and calculate SHAP values
mod_dt = joblib.load("artifacts/model.joblib")
if not os.path.exists("artifacts/shap_values.pkl"):
    print("Calculating SHAP values...")
    shap_values = shap.TreeExplainer(mod_dt).shap_values(X_train)
    joblib.dump(shap_values, "artifacts/shap_values.pkl")
else:
    print("Loading precomputed SHAP values...")
    shap_values = joblib.load("artifacts/shap_values.pkl")
    print("SHAP values loaded successfully.")
    print("SHAP values already exist. Skipping calculation.")
    print("SHAP values calculated and saved to artifacts/shap_values.pkl")

# Visualize SHAP values

shap.summary_plot(shap_values, X_train)

explainer = shap.KernelExplainer(mod_dt.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)
#shap.force_plot(explainer.expected_value[0], shap_values[..., 0], X_test)
# Create force plot
force_plot = shap.force_plot(explainer.expected_value[0], shap_values[..., 0], X_test)

# Save force plot to HTML
shap.save_html("artifacts/shap_force_plot.html", force_plot)