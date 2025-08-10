import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Load model and data
model = joblib.load("models/decision_tree_model.joblib")
X_test, y_test = joblib.load("models/test_data.joblib")

# Predict
y_pred = model.predict(X_test)

# Overall metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Classification report as dict
report_dict = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose().round(3)

# Create figure layout
fig = plt.figure(figsize=(18, 10))
grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

# --- Confusion Matrix ---
ax1 = fig.add_subplot(grid[0, 0])
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Blues, ax=ax1)
ax1.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

# --- Basic Metrics ---
ax2 = fig.add_subplot(grid[0, 1])
ax2.axis('off')
metrics_text = (
    f"Overall Metrics\n\n"
    f"Accuracy : {accuracy:.3f}\n"
    f"Precision: {precision:.3f}\n"
    f"Recall   : {recall:.3f}\n"
    f"F1 Score : {f1:.3f}"
)
ax2.text(0, 0.9, metrics_text, fontsize=14, va="top", ha="left", fontweight="bold")

# --- Per-class Metrics Table ---
ax3 = fig.add_subplot(grid[1, 0])
ax3.axis('off')
table_data = df_report.drop(index=["accuracy", "macro avg", "weighted avg"]).reset_index()
table_data.columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]
table = ax3.table(
    cellText=table_data.values,
    colLabels=table_data.columns,
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
ax3.set_title("Per-Class Metrics", fontsize=14, fontweight="bold")

# --- Bar Chart of Per-Class Metrics ---
ax4 = fig.add_subplot(grid[1, 1])
classes = table_data["Class"]
x = np.arange(len(classes))
width = 0.25
ax4.bar(x - width, table_data["Precision"], width, label="Precision")
ax4.bar(x, table_data["Recall"], width, label="Recall")
ax4.bar(x + width, table_data["F1-Score"], width, label="F1-Score")
ax4.set_xticks(x)
ax4.set_xticklabels(classes, rotation=45, ha="right")
ax4.set_ylim(0, 1)
ax4.set_ylabel("Score")
ax4.set_title("Per-Class Metrics Chart", fontsize=14, fontweight="bold")
ax4.legend()

# Adjust layout & save
plt.tight_layout()
os.makedirs("artifacts", exist_ok=True)
plt.savefig("artifacts/full_detailed_report.png", dpi=300)
plt.close()

print("Detailed evaluation dashboard saved to plots/full_detailed_report.png")
