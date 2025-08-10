import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

def plot_and_save_metrics():
    """
    Loads artifacts, evaluates the model, and plots and saves
    the confusion matrix and classification report as a single image.
    """
    print("--- Starting to Plot Metrics ---")

    # === Load data ===
    df = pd.read_csv("data/iris.csv")

    # === Load artifacts ===
    try:
        model = joblib.load("artifacts/model.joblib")
        le = joblib.load("artifacts/label_encoder.joblib")
        print("Artifacts loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure train.py has run and created artifacts.")
        return

    # === Prepare data dynamically based on model's expected features ===
    try:
        expected_features = model.feature_names_in_
        X = df[expected_features]
        y = df['species']
    except AttributeError:
        print("Warning: 'feature_names_in_' not found. Falling back to dropping columns.")
        X = df.drop(columns=['species', 'location'], errors='ignore')
        y = df['species']
    except KeyError as e:
        print(f"Error: Model was trained on features not present in the new data: {e}")
        return

    # === Re-create the exact same train/test split as in training ===
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )
    print(f"Test set created with {len(X_test)} samples.")

    # === Predict ===
    y_pred = model.predict(X_test)

    # === Generate Metrics for Plotting ===
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    report_df = pd.DataFrame(report_dict).transpose()

    # === Plot ===
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Performance Metrics", fontsize=16)

    # Plot 1: Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax[0], cmap="Blues", values_format='d')
    ax[0].set_title("Confusion Matrix")

    # Plot 2: Classification Report Table
    ax[1].axis("off") # Hide axes
    table = ax[1].table(
        cellText=report_df.round(2).values,
        rowLabels=report_df.index,
        colLabels=report_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    ax[1].set_title("Classification Report", pad=20)

    # === Save plot to artifacts directory ===
    os.makedirs("artifacts", exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("artifacts/metrics.png")
    
    print("Metrics plot saved to artifacts/metrics.png")
    print("----------------------------\n")

if __name__ == "__main__":
    plot_and_save_metrics()
