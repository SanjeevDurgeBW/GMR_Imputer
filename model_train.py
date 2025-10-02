import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from azure.storage.blob import BlobServiceClient
from pandas import concat
import os
from joblib import dump
from joblib import load
import joblib
import hashlib
from sequential_imputer import SequentialImputer
# from config import PRODUCT_MODEL_MAP, MODEL_DIR
import traceback
import logging
import subprocess
# import dbutils
import uuid
from io import BytesIO
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training_pipeline(product, uploaded_filepath, blob_connection_string, blob_container_name, train_percent=80):
    """
    Trains a SequentialImputer model for the given product and uploads it to Azure Blob Storage.
    """

    try:
        logger.info(f"Reading uploaded training file: {uploaded_filepath}")
        df = pd.read_excel(uploaded_filepath, engine='openpyxl')
        df = df.iloc[2:].reset_index(drop=True)
        df.columns = df.iloc[0]
        df = df.drop(df.index[0]).reset_index(drop=True)
        df = df.fillna("NONE")

        numerical_columns_as_categorical = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Numerical columns treated as categorical: {numerical_columns_as_categorical}")
        for col in numerical_columns_as_categorical:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2).astype(str)

        # Find the index of 'Forecast Date'
        all_columns = list(df.columns)
        try:
            forecast_index = all_columns.index('Forecast Date')
            initial_features = all_columns[:forecast_index + 1]
            columns_to_impute = all_columns[forecast_index + 1:]

            logger.info(f"Initial features used for imputation: {initial_features}")
            logger.info(f"Columns to impute: {columns_to_impute}")
        except ValueError:
            raise Exception("'Forecast Date' column not found in the dataset.")

        # Split into training and test sets based on RowStatus
        train_df = df[~df['RowStatus'].isin(["NEW", "AUTOFILL"])].copy()

        # Train/test split
        train_data, _ = train_test_split(train_df, test_size=(100 - train_percent) / 100, random_state=42)

        # Fit imputer
        imputer = SequentialImputer(columns_to_impute=columns_to_impute, initial_features=initial_features)
        imputer.fit(train_data)

        # # Save model locally
        # local_model_path = f"{product}_latest_model.pkl"
        # joblib.dump(imputer, local_model_path)
        # logger.info(f"Model saved locally at: {local_model_path}")

        # # Upload to Azure Blob Storage
        # blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        # blob_client = blob_service_client.get_blob_client(container=blob_container_name, blob=f"models/{product}/{product}_latest_model.pkl")

        # with open(local_model_path, "rb") as data:
        #     blob_client.upload_blob(data, overwrite=True)
        #     logger.info(f"Model uploaded to Azure Blob Storage: models/{product}/{product}_latest_model.pkl")

        # Optionally delete local file
        # os.remove(local_model_path)

        # Generate versioned filename
        # timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        # unique_id = uuid.uuid4().hex[:8]
        # versioned_filename = f"{product}_model_{timestamp}_{unique_id}.pkl"
        # local_model_path = versioned_filename
        # joblib.dump(imputer, local_model_path)
        # logger.info(f"Model saved locally at: {local_model_path}")

        # # Upload to Azure Blob Storage with metadata
        # blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        # blob_client = blob_service_client.get_blob_client(
        #     container=blob_container_name,
        #     blob=f"models/{product}/{versioned_filename}"
        # )

        # metadata = {
        #     "product": product,
        #     "trained_on": timestamp,
        #     "model_type": "SequentialImputer",
        #     "version": unique_id
        # }

        # with open(local_model_path, "rb") as data:
        #     blob_client.upload_blob(data, overwrite=True, metadata=metadata)
        #     logger.info(f"Model uploaded to Azure Blob Storage: models/{product}/{versioned_filename}")

        # os.remove(local_model_path)

        versioned_filename = upload_model_with_checksum_to_blob(
            model_object=imputer,
            product=product,
            blob_connection_string=blob_connection_string,
            blob_container_name=blob_container_name
        )

        # upload_model_and_checksum_to_unity_catalog(
        #     model_object=imputer,
        #     product="cv_egr",
        #     volume_path="/Volumes/ml_catalog/ml_schema/model_volume"
        # )

        # save_model_and_checksum_locally(
        #     model_object=imputer,    
        #     product=product,
        #     output_dir="models"  # Local directory to save models
        # )
        logger.info(f"Model for '{product}' successfully trained and uploaded.")

        return versioned_filename

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}\n{traceback.format_exc()}")
        raise


def upload_model_with_checksum_to_blob(model_object, product, blob_connection_string, blob_container_name):
    """
    Serializes model in memory, computes SHA256 checksum, and uploads both to Azure Blob Storage.
    """

    # Generate versioned filename
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    versioned_filename = f"{product}_model_{timestamp}_{unique_id}.pkl"
    checksum_filename = versioned_filename.replace(".pkl", ".pkl.sha256")

    # Serialize model to memory
    model_buffer = BytesIO()
    joblib.dump(model_object, model_buffer)
    model_bytes = model_buffer.getvalue()

    # Compute SHA256 checksum
    # checksum = hashlib.sha256(model_bytes).hexdigest()
    sha256_hash = hashlib.sha256()
    for byte_block in iter(lambda: model_buffer.read(4096), b""):
        sha256_hash.update(byte_block)
    checksum = sha256_hash.hexdigest()
    checksum_content = f"{checksum}  {versioned_filename}".encode("utf-8")

    # Upload both blobs
    blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
    model_blob = blob_service_client.get_blob_client(container=blob_container_name, blob=f"models/{product}/{versioned_filename}")
    checksum_blob = blob_service_client.get_blob_client(container=blob_container_name, blob=f"models/{product}/{checksum_filename}")

    model_blob.upload_blob(model_bytes, overwrite=True)
    checksum_blob.upload_blob(checksum_content, overwrite=True)

    logger.info(f"Uploaded model: models/{product}/{versioned_filename}")
    logger.info(f"Uploaded checksum: models/{product}/{checksum_filename}")

    return versioned_filename

def upload_model_and_checksum_to_unity_catalog(model_object, product, volume_path):
    """
    Uploads model and checksum to Unity Catalog volume in Databricks.
    volume_path example: '/Volumes/ml_catalog/ml_schema/model_volume'
    """

    # Generate versioned filenames
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    model_filename = f"{product}_model_{timestamp}_{unique_id}.pkl"
    checksum_filename = model_filename.replace(".pkl", ".pkl.sha256")

    # Serialize model to memory
    model_buffer = BytesIO()
    joblib.dump(model_object, model_buffer)
    model_bytes = model_buffer.getvalue()

    # Compute SHA256 checksum
    sha256_hash = hashlib.sha256(model_bytes).hexdigest()
    checksum_content = f"{sha256_hash}  {model_filename}"

    # Save model to temp file and upload to Unity Catalog
    with open(f"/tmp/{model_filename}", "wb") as f:
        f.write(model_bytes)
    dbutils.fs.cp(f"file:/tmp/{model_filename}", f"dbfs:{volume_path}/{model_filename}")

    # Save checksum to temp file and upload
    with open(f"/tmp/{checksum_filename}", "w") as f:
        f.write(checksum_content)
    dbutils.fs.cp(f"file:/tmp/{checksum_filename}", f"dbfs:{volume_path}/{checksum_filename}")

    print(f"✅ Model uploaded to: {volume_path}/{model_filename}")
    print(f"✅ Checksum uploaded to: {volume_path}/{checksum_filename}")

    return model_filename

# def save_model_and_checksum_locally(model_object, product, output_dir="models"):
#     """
#     Saves model and its SHA256 checksum to local disk with versioned filenames.
#     """

#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Generate versioned filename
#     timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
#     unique_id = uuid.uuid4().hex[:8]
#     model_filename = f"{product}_model_{timestamp}_{unique_id}.pkl"
#     model_path = os.path.join(output_dir, model_filename)

#     # Save model
#     joblib.dump(model_object, model_path)
#     print(f"Model saved to: {model_path}")

#     # Compute SHA256 checksum
#     sha256_hash = hashlib.sha256()
#     with open(model_path, "rb") as f:
#         for byte_block in iter(lambda: f.read(4096), b""):
#             sha256_hash.update(byte_block)
#     checksum = sha256_hash.hexdigest()

#     # Save checksum file
#     sha_filename = model_filename.replace(".pkl", ".pkl.sha256")
#     sha_path = os.path.join(output_dir, sha_filename)
#     with open(sha_path, "w") as sha_file:
#         sha_file.write(f"{checksum}  {model_filename}")
#     print(f"SHA256 checksum saved to: {sha_path}")

#     return model_path, sha_path

# product = "cv_egr"
# uploaded_filepath = "C:\\Users\\sdurge\Downloads\\CV EGR System Lookups - 03-2025.xlsx"
# blob_connection_string = r"DefaultEndpointsProtocol=https;AccountName=youraccount;AccountKey=yourkey;EndpointSuffix=core.windows.net"
# blob_container_name = "ml-models"

# result = run_training_pipeline(
#     product=product,
#     uploaded_filepath=uploaded_filepath,
#     blob_connection_string=blob_connection_string,
#     blob_container_name=blob_container_name
# )

# print(result)



# @app.route('/train', methods=['POST'])
# def train():
#     product = request.form.get('product')
#     train_file = request.files.get('train_file')
#     # train_percent = int(request.form.get('train_percent', 80))

#     if not product or not train_file:
#         flash("Please select a product and upload training data.", "danger")
#         return redirect(url_for('home'))

#     try:
#         # Save uploaded file
#         filename = secure_filename(train_file.filename)
#         filepath = os.path.join("uploads", filename)
#         train_file.save(filepath)

#         # Run training logic here
#         accuracy_file = run_training_pipeline(product, filepath)

#         flash(f"Training complete for {product}.", "success")
#         return render_template('index.html', training_complete=True, accuracy_file=accuracy_file)

#     except Exception as e:
#         logger.error(f"Training failed: {e}")
#         flash(f"Training failed: {str(e)}", "danger")
#         return redirect(url_for('home'))

# @app.route('/download/<filename>')
# def download_accuracy(filename):
#     return send_from_directory('metrics', filename, as_attachment=True)

