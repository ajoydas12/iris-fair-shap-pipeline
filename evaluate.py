import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Load model and test data
model = joblib.load("models/decision_tree_model.joblib")
X_test, y_test = joblib.load("models/test_data.joblib")

# Predict
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # 'macro' for multiclass
f1 = f1_score(y_test, y_pred, average='macro')

# Print metrics
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"F1 Score:  {f1:.3f}")

# Save plots and reports
os.makedirs("plots", exist_ok=True)

# Confusion Matrix
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Blues)
disp.ax_.set_title("Confusion Matrix")
plt.savefig("plots/confusion_matrix.png")
plt.close()

# Classification Report
report = classification_report(y_test, y_pred)
with open("plots/classification_report.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(report)

# Save basic metrics to file
with open("plots/metrics.txt", "w") as f:
    f.write(f"Accuracy:  {accuracy:.3f}\n")
    f.write(f"Precision: {precision:.3f}\n")
    f.write(f"F1 Score:  {f1:.3f}\n")

print("Metrics saved to plots/metrics.txt")
print("Confusion matrix saved to plots/confusion_matrix.png")
print("Classification report saved to plots/classification_report.txt")
