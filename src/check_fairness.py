import pandas as pd
import joblib
import json
import os
from fairlearn.metrics import MetricFrame, demographic_parity_difference


def check_model_fairness():
    """
    Loads the trained model and assesses its fairness for all classes based 
    on the 'location' sensitive feature.
    """
    print("--- Checking Model Fairness ---")

    try:
        # Load model and data
        model = joblib.load("artifacts/decision_tree_model.joblib")
        df = pd.read_csv("data/iris.csv")

        if 'location' not in df.columns:
            print("❌ Error: 'location' column not found in data. Please run induce_bias.py first.")
            return

        # Get features the model was trained on
        expected_features = getattr(
            model, 'feature_names_in_',
            df.drop(columns=['species', 'location']).columns
        )

        X = df[expected_features]
        y_true = df['species']
        sensitive_feature = df['location']

        print("✅ Model and data with 'location' feature loaded.")
    except (FileNotFoundError, AttributeError, KeyError, Exception) as e:
        print(f"❌ Error during data/model loading: {e}")
        return

    # Predict with the model
    y_pred = model.predict(X)

    # MetricFrame metrics (selection_rate is for 'versicolor' as an example)
    metrics = {
        'accuracy': lambda y_true, y_pred: (y_true == y_pred).mean(),
        'selection_rate_versicolor': lambda y_true, y_pred: (y_pred == 'Versicolor').mean()
    }

    grouped_metrics = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )

    print("\n📊 Fairness metrics by 'location' group:")
    print(grouped_metrics.by_group)

    # Compute demographic parity difference for each class
    fairness_report = {}
    print("\n📏 Calculating Demographic Parity Difference per class...")
    for cls in model.classes_:
        metric_name = f"demographic_parity_difference_{cls}"

        # Manually create binary arrays for compatibility
        y_true_binary = (y_true == cls)
        y_pred_binary = (y_pred == cls)

        dpd = demographic_parity_difference(
            y_true_binary,
            y_pred_binary,
            sensitive_features=sensitive_feature
        )
        fairness_report[metric_name] = dpd

    print("\n✅ Overall Fairness Report:")
    print(json.dumps(fairness_report, indent=2))

    # Save report to JSON file
    os.makedirs("artifacts", exist_ok=True)
    report_path = "artifacts/fairness_report.json"
    with open(report_path, "w") as f:
        json.dump(fairness_report, f, indent=4)

    print(f"\n💾 Fairness report saved to: {report_path}")
    print("-----------------------------\n")


if __name__ == "__main__":
    check_model_fairness()
