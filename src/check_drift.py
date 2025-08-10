from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset 
import os
import pandas as pd

data = pd.read_csv('data/iris.csv')
# Map column types for evidently
schema = DataDefinition(
    numerical_columns=["sepal_length","sepal_width","petal_length","petal_width"],
    categorical_columns=["species", "location"],
    )

# Intentionally create unseen data
new_data = data.copy()
noise_data = new_data[new_data['sepal_length']>7.5].reset_index(drop=True)
noise_data['sepal_length']=15.0
noise_data['species']='fakeiris'
noise_data['petal_length']=100.0
new_data = pd.concat([new_data,noise_data], ignore_index=True)
new_data

# Create evidently data sets, with original data as reference and new data for comparison
eval_orig_data = Dataset.from_pandas(data, data_definition=schema)
eval_new_data = Dataset.from_pandas(new_data, data_definition=schema)

# Run drift report
report = Report([
    DataDriftPreset(),
    DataSummaryPreset()
],
include_tests='True')

my_eval = report.run(eval_new_data, eval_orig_data)
my_eval

# Save the report
if not os.path.exists("artifacts"):
    os.makedirs("artifacts")

my_eval.save_html("artifacts/drift_report.html")
print("Drift report saved to artifacts/drift_report.html")
print("Drift analysis completed successfully. ✓")