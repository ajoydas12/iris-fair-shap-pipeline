import os, joblib, mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from mlflow.models import infer_signature
from google.cloud import aiplatform, storage

print("DEMO")
# --- Configuration ---
# In a real pipeline, these would come from environment variables or a config file
PROJECT_ID = "premium-cipher-462011-p3"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
BUCKET_URI = f"gs://mlops-course-premium-cipher-462011-p3-unique"  # @param {type:"string"}

MODEL_ARTIFACT_DIR = "my-models/iris-classifier-week-1"  # @param {type:"string"}
REPOSITORY = "iris-classifier-repo"  # @param {type:"string"}
IMAGE = "iris-classifier-img"  # @param {type:"string"}
MODEL_DISPLAY_NAME = "iris-classifier"  # @param {type:"string"}


# --- MLflow Tracking URI based on environment ---
if os.getenv('CI'):
    # In CI, save MLflow data locally
    mlflow_tracking_uri = "file:./mlruns"
    print(f"CI environment detected. Using local MLflow tracking URI: {mlflow_tracking_uri}")
    REGISTERED_MODEL_NAME = ""
else:
    # Replace with the actual external IP of your GCP instance
    EXTERNAL_IP = "http://34.59.44.241:8100"  # Replace this dynamically if needed
    mlflow_tracking_uri = EXTERNAL_IP
    print(f"Local environment detected. Using remote MLflow tracking URI: {mlflow_tracking_uri}")
    REGISTERED_MODEL_NAME = "IRIS-classifier-decisiontrees"

# --- Initialize clients and MLflow ---
aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Iris_Classification_Experiment")

# --- Helper Function for GCS Upload ---
def upload_to_gcs(bucket_name, source_file_path, destination_blob_name):
    """Uploads a file to the specified GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)
    print(f"File {source_file_path} uploaded to gs://{bucket_name}/{destination_blob_name}")
# Load data
data = pd.read_csv("data/iris.csv")
X = data[['sepal_length','sepal_width','petal_length','petal_width']]
y = data['species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Train model
print("Fitting LabelEncoder...")
le = LabelEncoder()
le.fit(y_train)

# 4. Train Model
params = {
    "max_depth": 4,
    "random_state": 1
}
model = DecisionTreeClassifier(**params)
model.fit(X_train, y_train)

# # Save model and test data
# os.makedirs("models", exist_ok=True)
# joblib.dump(model, "models/decision_tree_model.joblib")
# joblib.dump((X_test, y_test), "models/test_data.joblib")

# print("Model and test data saved.")



# 5. Evaluate Model
prediction = model.predict(X_test)
accuracy_score = metrics.accuracy_score(prediction, y_test)
print('The accuracy of the Decision Tree is', "{:.3f}".format(accuracy_score))

# 6. Save and Upload Artifacts (Model AND Encoder)
os.makedirs("artifacts", exist_ok=True)
print("Saving model and label encoder artifacts...")
joblib.dump(model, "artifacts/model.joblib")
joblib.dump(le, "artifacts/label_encoder.joblib") ### FIXED ###: Save the encoder

# Use the Python function for GCS upload
bucket_name_str = BUCKET_URI.replace("gs://", "")
model_gcs_path = f"{MODEL_ARTIFACT_DIR}/model.joblib"
encoder_gcs_path = f"{MODEL_ARTIFACT_DIR}/label_encoder.joblib" ### FIXED ###: Define GCS path for encoder

upload_to_gcs(bucket_name_str, "artifacts/model.joblib", model_gcs_path)
upload_to_gcs(bucket_name_str, "artifacts/label_encoder.joblib", encoder_gcs_path) ### FIXED ###: Upload the encoder

# 7. Log Experiment with MLflow
with mlflow.start_run() as run:
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy_score)
    mlflow.set_tag("Training Info", "Decision tree model for IRIS data")

    signature = infer_signature(X_train, model.predict(X_train))
    
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train.head(1),
        # Use the variable to conditionally register the model
        registered_model_name=REGISTERED_MODEL_NAME if REGISTERED_MODEL_NAME else None, ### FIXED ###
    )
    print(f"MLflow Run completed. Run ID: {run.info.run_id}")
