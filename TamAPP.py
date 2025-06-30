import os
import joblib
from flask import Flask, request, send_file, render_template, redirect, url_for, flash, jsonify
import pandas as pd
import logging
import traceback
import sys
import hashlib
import asyncio  # For asynchronous operations
import aiofiles  # For asynchronous file operations

from azure.storage.blob.aio import BlobServiceClient  # Asynchronous BlobServiceClient
from azure.core.exceptions import AzureError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# This is the same class you used when training
from sequential_imputer import SequentialImputer

# ----------------------------------------------------------------
# Configure Logging
# ----------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Create the Flask app
# ----------------------------------------------------------------
app = Flask(__name__)

# ----------------------------------------------------------------
# Log the SECRET_KEY from environment
# ----------------------------------------------------------------
logger.info("Checking environment for SECRET_KEY...")
secret_from_env = os.environ.get("SECRET_KEY")
logger.info(f"ENV SECRET_KEY: {secret_from_env}")

# If environment variable is missing, we fall back to 'hello'
app.secret_key = secret_from_env or "hello"
logger.info(f"app.secret_key has been set to: {app.secret_key}")

# ----------------------------------------------------------------
# Global Paths
# ----------------------------------------------------------------
HOME_DIR = os.environ.get('HOME', '/home')  # On Azure App Service, /home is persistent
MODEL_DIR = os.path.join(HOME_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'Tam_eheatingV5.pkl')
CHECKSUM_PATH = os.path.join(MODEL_DIR, 'Tam_eheatingV5.pkl.sha256')

# ----------------------------------------------------------------
# Lock and Global Model Pipeline
# ----------------------------------------------------------------
model_lock = asyncio.Lock()
model_pipeline = None  # will store the loaded SequentialImputer

# ----------------------------------------------------------------
# Ensure Model Directory
# ----------------------------------------------------------------
def ensure_model_directory():
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info(f"Model directory is ready at '{MODEL_DIR}'.")
    except Exception as e:
        logger.error(f"Error ensuring model directory exists: {e}\n{traceback.format_exc()}")
        raise

# ----------------------------------------------------------------
# Checksum Utilities
# ----------------------------------------------------------------
def compute_sha256(file_path):
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error computing SHA-256 checksum for '{file_path}': {e}\n{traceback.format_exc()}")
        raise

def verify_checksum(model_path, checksum_path):
    try:
        if not os.path.exists(checksum_path):
            raise FileNotFoundError(f"Checksum file '{checksum_path}' not found.")

        with open(checksum_path, 'r') as f:
            expected_checksum = f.read().split()[0].strip()

        actual_checksum = compute_sha256(model_path)

        logger.info(f"Expected Checksum (SHA-256): {expected_checksum}")
        logger.info(f"Actual Checksum  (SHA-256): {actual_checksum}")

        if actual_checksum.lower() != expected_checksum.lower():
            raise ValueError(
                f"Checksum mismatch for '{model_path}'. "
                f"Expected {expected_checksum}, got {actual_checksum}."
            )

        logger.info(f"Checksum verification passed for '{model_path}'.")
    except Exception as e:
        logger.error(f"Error verifying checksum for '{model_path}': {e}\n{traceback.format_exc()}")
        raise

# ----------------------------------------------------------------
# Async Download with Retry
# ----------------------------------------------------------------
@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((EnvironmentError, FileNotFoundError, AzureError))
)
async def async_download_blob(container_name, blob_name, download_file_path):
    try:
        os.makedirs(os.path.dirname(download_file_path), exist_ok=True)

        connect_str = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
        if not connect_str:
            raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING is not set.")

        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        exists = await blob_client.exists()
        if not exists:
            raise FileNotFoundError(f"Blob '{blob_name}' not found in container '{container_name}'.")

        logger.info(f"Starting download of blob '{blob_name}' from container '{container_name}'.")
        async with aiofiles.open(download_file_path, "wb") as download_file:
            download_stream = await blob_client.download_blob()
            async for chunk in download_stream.chunks():
                await download_file.write(chunk)

        logger.info(f"Downloaded blob '{blob_name}' to '{download_file_path}'.")
    except Exception as e:
        logger.error(f"Error downloading blob '{blob_name}': {e}\n{traceback.format_exc()}")
        raise

# ----------------------------------------------------------------
# Download & Verify Model
# ----------------------------------------------------------------
async def async_download_and_verify_model():
    try:
        # 1) Download the .pkl
        await async_download_blob(
            container_name='models',
            blob_name='Tam_eheatingV5.pkl',
            download_file_path=MODEL_PATH
        )
        # 2) Download the .sha256
        await async_download_blob(
            container_name='models',
            blob_name='Tam_eheatingV5.pkl.sha256',
            download_file_path=CHECKSUM_PATH
        )
        # 3) Verify
        verify_checksum(MODEL_PATH, CHECKSUM_PATH)
    except Exception as e:
        logger.error(f"Error in async_download_and_verify_model: {e}\n{traceback.format_exc()}")
        raise

# ----------------------------------------------------------------
# Load the Trained Imputer at Startup
# ----------------------------------------------------------------
async def async_load_model_pipeline():
    global model_pipeline
    if model_pipeline is not None:
        logger.info("Model pipeline already loaded. Skipping download.")
        return

    async with model_lock:
        if model_pipeline is not None:
            logger.info("Model pipeline already loaded inside lock. Skipping download.")
            return
        try:
            ensure_model_directory()
            logger.info("Starting asynchronous model download and verification...")
            await async_download_and_verify_model()

            logger.info(f"Loading model from '{MODEL_PATH}'...")
            loop = asyncio.get_event_loop()
            model_pipeline = await loop.run_in_executor(None, joblib.load, MODEL_PATH)
            logger.info("Model pipeline loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}\n{traceback.format_exc()}")
            raise

# ----------------------------------------------------------------
# Flask Routes
# ----------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    """
    Health check endpoint for Azure App Service.
    """
    return jsonify({"status": "OK"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    This endpoint:
      1) Receives an Excel file
      2) Reads it into a DataFrame (using openpyxl engine)
      3) Preprocesses (so that columns align with the trained imputer)
      4) Calls model_pipeline.transform(...) to fill missing/impute columns
      5) Returns the imputed DataFrame as 'predictions.xlsx'
    """
    if 'file' not in request.files:
        logger.warning("No file part in the request.")
        flash("No file part in the request.", "danger")
        return redirect(url_for('home'))

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        logger.warning("No selected file.")
        flash("No selected file.", "danger")
        return redirect(url_for('home'))

    if not allowed_file(uploaded_file.filename):
        logger.warning("Unsupported file type.")
        flash("Unsupported file type. Please upload an Excel file (.xls or .xlsx).", "danger")
        return redirect(url_for('home'))

    try:
        # 1) Read Excel file into a DataFrame
        input_data = pd.read_excel(uploaded_file, engine="openpyxl")
        logger.info("Received Excel file for imputation.")
        logger.info(f"RAW columns from uploaded Excel: {list(input_data.columns)}")

        # 2) Preprocess input_data so columns match exactly what the imputer expects
        input_data = preprocess_input_data(input_data)
        logger.info("Preprocessed input data to match training.")
        logger.info(f"After preprocessing, columns are: {list(input_data.columns)}")

        # 3) Check if model pipeline is loaded
        if model_pipeline is None:
            logger.warning("Model pipeline not loaded in memory. Aborting.")
            flash("Model pipeline not loaded in memory.", "danger")
            return redirect(url_for('home'))

        # 4) Transform the data to fill missing columns
        logger.info("Imputing data with model_pipeline.transform(...)")
        imputed_data = model_pipeline.transform(input_data)
        logger.info("Data imputation completed successfully.")

        # 5) Save the imputed DataFrame as 'predictions.xlsx'
        output_file = 'predictions.xlsx'
        imputed_data.to_excel(output_file, index=False)

        logger.info(f"Imputed data saved to '{output_file}'.")

        # 6) Return the Excel file to the user
        return send_file(
            output_file,
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        error_message = f"Error during imputation: {e}\n{traceback.format_exc()}"
        logger.error(error_message)
        flash(f"An error occurred during imputation: {str(e)}", "danger")
        return redirect(url_for('home'))

# ----------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------
def allowed_file(filename):
    """
    Checks if the uploaded file is an Excel file (.xls or .xlsx).
    """
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ["xls", "xlsx"]

def preprocess_input_data(df):
    """
    IMPORTANT:
    Replicate the same cleaning steps you did locally before training.
    For example:
       df = df.iloc[2:].reset_index(drop=True)
       df.columns = df.iloc[0]
       df = df.drop(df.index[0]).reset_index(drop=True)
       df = df.iloc[:, :-17]
       df = df.fillna("NONE")

    We'll do exactly that here to ensure your columns
    (like 'Cabin Heat Technology') exist and match
    the names your imputer was trained on.
    """
    logger.info(f"BEFORE custom slicing, columns are: {list(df.columns)} (rows={df.shape[0]})")

    # Skip the first two rows
    df = df.iloc[2:].reset_index(drop=True)

    # Let the next row become the column headers
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True)

    logger.info(f"AFTER making row 0 a header, columns are: {list(df.columns)} (rows={df.shape[0]})")

    # Drop the last 17 columns
    logger.info("Dropping the last 17 columns...")
    df = df.iloc[:, :-17]

    # Fill missing values with "NONE"
    df = df.fillna("NONE")

    logger.info(f"AFTER dropping last 17 cols and fillna, columns: {list(df.columns)} (rows={df.shape[0]})")

    return df

# ----------------------------------------------------------------
# Run the Flask App
# ----------------------------------------------------------------
if __name__ == "__main__":
    try:
        ensure_model_directory()
        logger.info("Loading model pipeline once at startup...")
        asyncio.run(async_load_model_pipeline())
        app.run(host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Application failed to start: {e}\n{traceback.format_exc()}")
        sys.exit(1)