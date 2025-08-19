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
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
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

#---------------------------------------------------
# Products Model Map
#---------------------------------------------------
# This is a dictionary mapping product names to their respective model classes.
HOME_DIR = os.environ.get('ROOT', '/root')  # On Azure App Service, /root is persistent
# HOME_DIR = os.environ.get('HOME', '/home')
MODEL_DIR = os.path.join(HOME_DIR, 'models')
PRODUCT_MODEL_MAP = {
    "lv_turbo": {
        "model_path": os.path.join(MODEL_DIR, "lv_turbo_model.pkl"),
        # "compressed_model_path": os.path.join(MODEL_DIR, "lv_turbo_model.pkl.gz"),
        "checksum_path": os.path.join(MODEL_DIR, "lv_turbo_model.pkl.sha256"),
        "blob_model": "lv_turbo_model.pkl",
        "blob_checksum": "lv_turbo_model.pkl.sha256",
        "drop_cols": 19
    },
    "lv_egr": {
        "model_path": os.path.join(MODEL_DIR, "lv_egr_model.pkl"),
        # "compressed_model_path": os.path.join(MODEL_DIR, "lv_egr_model.pkl.gz"),
        "checksum_path": os.path.join(MODEL_DIR, "lv_egr_model.pkl.sha256"),
        "blob_model": "lv_egr_model.pkl",
        "blob_checksum": "lv_egr_model.pkl.sha256",
        "drop_cols": 19
    },
    "cv_turbo": {
        "model_path": os.path.join(MODEL_DIR, "cv_turbo_model.pkl"),
        # "compressed_model_path": os.path.join(MODEL_DIR, "cv_turbo_model.pkl.gz"),
        "checksum_path": os.path.join(MODEL_DIR, "cv_turbo_model.pkl.sha256"),
        "blob_model": "cv_turbo_model.pkl",
        "blob_checksum": "cv_turbo_model.pkl.sha256",
        "drop_cols": 23
    },
    "cv_egr": {
        "model_path": os.path.join(MODEL_DIR, "cv_egr_model.pkl"),
        # "compressed_model_path": os.path.join(MODEL_DIR, "cv_egr_model.pkl.gz"),
        "checksum_path": os.path.join(MODEL_DIR, "cv_egr_model.pkl.sha256"),
        "blob_model": "cv_egr_model.pkl",
        "blob_checksum": "cv_egr_model.pkl.sha256",
        "drop_cols": 18
    },
    "tam_eheating": {
        "model_path": os.path.join(MODEL_DIR, "tam_eheating_model.pkl"),
        # "compressed_model_path": os.path.join(MODEL_DIR, "tam_eheating_model.pkl.gz"),
        "checksum_path": os.path.join(MODEL_DIR, "tam_eheating_model.pkl.sha256"),
        "blob_model": "tam_eheating_model.pkl",
        "blob_checksum": "tam_eheating_model.pkl.sha256",
        "drop_cols": 19
    }    
}


# ----------------------------------------------------------------
# Lock and Global Model Pipeline
# ----------------------------------------------------------------
model_lock = asyncio.Lock()
model_pipeline = None  # will store the loaded SequentialImputer
# model_pipeline = {}  # product -> pipeline

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
# -------------------------new code------------------------

# @retry(
#     reraise=True,
#     stop=stop_after_attempt(5),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type((EnvironmentError, FileNotFoundError, AzureError))
# )
# async def async_download_blob(container_name, blob_name, download_file_path):
#     try:
#         os.makedirs(os.path.dirname(download_file_path), exist_ok=True)

#         connect_str = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
#         if not connect_str:
#             raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING is not set.")

#         # Use async context manager to ensure session is closed
#         async with BlobServiceClient.from_connection_string(connect_str) as blob_service_client:
#             blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

#             exists = await blob_client.exists()
#             if not exists:
#                 raise FileNotFoundError(f"Blob '{blob_name}' not found in container '{container_name}'.")

#             logger.info(f"Starting download of blob '{blob_name}' from container '{container_name}'.")
#             async with aiofiles.open(download_file_path, "wb") as download_file:
#                 download_stream = await blob_client.download_blob()
#                 async for chunk in download_stream.chunks():
#                     await download_file.write(chunk)

#         logger.info(f"Downloaded blob '{blob_name}' to '{download_file_path}'.")
#     except Exception as e:
#         logger.error(f"Error downloading blob '{blob_name}': {e}\n{traceback.format_exc()}")
#         raise
# -------------------------new code------------------------


# ----------------------------------------------------------------
# Download & Verify Model
# ----------------------------------------------------------------
#---------------------new code------------------
async def async_download_and_verify_model(product):
    try:
        product_model_map = PRODUCT_MODEL_MAP[product]
        await async_download_blob(
            container_name='models',
            blob_name=product_model_map["blob_model"],
            download_file_path=product_model_map["model_path"]
        )
        await async_download_blob(
            container_name='models',
            blob_name=product_model_map["blob_checksum"],
            download_file_path=product_model_map["checksum_path"]
        )
        verify_checksum(product_model_map["model_path"], product_model_map["checksum_path"])
    except Exception as e:
        logger.error(f"Error in async_download_and_verify_model: {e}\n{traceback.format_exc()}")
        raise
# --------------new code--------------------------

def sync_download_and_verify_model(product):
    """
    Synchronously download and verify the model from Azure Blob Storage.
    """
    import asyncio
    try:
        asyncio.run(async_download_and_verify_model(product))
    except Exception as e:
        logger.error(f"Failed to download/verify model at startup: {e}\n{traceback.format_exc()}")
        raise


def sync_load_model_pipeline(product):
    """
    Synchronously load the model pipeline into memory.
    """
    global model_pipeline
    product_model_map = PRODUCT_MODEL_MAP[product]
    try:
        ensure_model_directory()
        if not os.path.exists(product_model_map['model_path']):
            logger.error(f"Model file not found at {product_model_map['model_path']}")
            raise FileNotFoundError(f"Model file not found at {product_model_map['model_path']}")
        model_pipeline = joblib.load(product_model_map['model_path'])
        logger.info("Model pipeline loaded into memory.")
    except Exception as e:
        logger.error(f"Failed to load model pipeline: {e}\n{traceback.format_exc()}")
        raise
# ----------------------------------------------------------------
# Load the Trained Imputer at Startup
# ----------------------------------------------------------------
async def async_load_model_pipeline(product):
    global model_pipeline
    product_model_map = PRODUCT_MODEL_MAP[product]
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
            await async_download_and_verify_model(product)

            logger.info(f"Loading model from '{product_model_map['model_path']}'...")
            loop = asyncio.get_event_loop()
            model_pipeline = await loop.run_in_executor(None, joblib.load, product_model_map['model_path'])
            logger.info("Model pipeline loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}\n{traceback.format_exc()}")
            raise


# async def async_load_model_pipeline(product):
#     global model_pipeline
#     product_model_map = PRODUCT_MODEL_MAP[product]
#     if product in model_pipeline and model_pipeline[product] is not None:
#         logger.info(f"Model pipeline for '{product}' already loaded. Skipping download.")
#         return

#     async with model_lock:
#         if product in model_pipeline and model_pipeline[product] is not None:
#             logger.info(f"Model pipeline for '{product}' already loaded inside lock. Skipping download.")
#             return
#         try:
#             ensure_model_directory()
#             logger.info("Starting asynchronous model download and verification...")
#             await async_download_and_verify_model(product)

#             logger.info(f"Loading model from '{product_model_map['model_path']}'...")
#             loop = asyncio.get_event_loop()
#             model_pipeline[product] = await loop.run_in_executor(None, joblib.load, product_model_map['model_path'])
#             logger.info("Model pipeline loaded successfully.")
#         except Exception as e:
#             logger.error(f"Error loading model: {e}\n{traceback.format_exc()}")
#             raise

# ----------------------------------------------------------------
# Flask Routes
# ----------------------------------------------------------------
@app.route('/')
def home():
    # return render_template('index.html')
    return render_template('index.html', products=PRODUCT_MODEL_MAP.keys())

@app.route('/health')
def health():
    """
    Health check endpoint for Azure App Service.
    """
    return jsonify({"status": "OK"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    product = request.form.get('product')
    if not product or product not in PRODUCT_MODEL_MAP:
        flash("Invalid product selected.", "danger")
        return redirect(url_for('home'))

    product_model_map = PRODUCT_MODEL_MAP[product]

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
        # Download and verify the model for the selected product
        # asyncio.run(async_download_and_verify_model(product))

        # # Load the model for the selected product
        # model_pipeline = joblib.load(product_model_map["model_path"])

        try:
            ensure_model_directory()
            logger.info("Downloading and verifying model...")
            sync_download_and_verify_model(product)
            logger.info("Loading model pipeline...")
            sync_load_model_pipeline(product)
        except Exception as e:
            logger.error(f"Application failed to load model: {e}\n{traceback.format_exc()}")

        # Ensure model is loaded and verified for this product (async, per-product)
        # await async_load_model_pipeline(product)
        # asyncio.run(async_load_model_pipeline(product))
        # pipeline = model_pipeline.get(product)
        # if pipeline is None:
        #     logger.warning("Model pipeline not loaded in memory. Aborting.")
        #     flash("Model pipeline not loaded in memory.", "danger")
        #     return redirect(url_for('home'))

        # Read Excel file into a DataFrame
        if product == "cv_turbo":
            input_data = pd.read_excel(uploaded_file, sheet_name='Data', engine="openpyxl")
        input_data = pd.read_excel(uploaded_file, engine="openpyxl")
        logger.info("Received Excel file for imputation.")
        logger.info(f"RAW columns from uploaded Excel: {list(input_data.columns)}")

        # Preprocess input_data so columns match exactly what the imputer expects
        drop_cols = product_model_map.get("drop_cols", 0)
        input_data = preprocess_input_data(input_data, drop_cols)
        logger.info("Preprocessed input data to match training.")
        logger.info(f"After preprocessing, columns are: {list(input_data.columns)}")

        # ----------------------------------------- NEW CODE -----------------------------------------
        # try:
        #     ensure_model_directory()
        #     logger.info("Downloading and verifying model...")
        #     sync_download_and_verify_model(product)
        #     logger.info("Loading model pipeline...")
        #     sync_load_model_pipeline(product)
        # except Exception as e:
        #     logger.error(f"Application failed to load model: {e}\n{traceback.format_exc()}")


        # try:
        #     ensure_model_directory()
        #     logger.info("Downloading and verifying model...")
        #     asyncio.run(async_load_model_pipeline(product))
        #     logger.info("Loading model pipeline...")
        # except Exception as e:
        #     logger.error(f"Application failed to load model: {e}\n{traceback.format_exc()}")
# ----------------------------------------- NEW CODE -----------------------------------------

        # 3) Check if model pipeline is loaded
        if model_pipeline is None:
            logger.warning("Model pipeline not loaded in memory. Aborting.")
            flash("Model pipeline not loaded in memory.", "danger")
            return redirect(url_for('home'))

        # Impute data
        logger.info("Imputing data with model_pipeline.transform(...)")
        imputed_data = model_pipeline.transform(input_data)
        logger.info("Data imputation completed successfully.")
# ------------new code ---------------------
        # loop = asyncio.get_running_loop()
        # imputed_data = await loop.run_in_executor(None, pipeline.transform, input_data)
        # logger.info("Data imputation completed successfully.")
# ------------new code ---------------------

        # # Save the imputed DataFrame as 'predictions.xlsx'
        # output_file = 'predictions.xlsx'
        # imputed_data.to_excel(output_file, index=False)
        # logger.info(f"Imputed data saved to '{output_file}'.")

        # Save the imputed DataFrame with a dynamic filename based on product
        output_file = f'{product}_predictions.xlsx'
        imputed_data.to_excel(output_file, index=False)
        logger.info(f"Imputed data saved to '{output_file}'.")

        # Return the Excel file to the user
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
#-------------------new code -------------------------

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

def preprocess_input_data(df, drop_cols):
    logger.info(f"BEFORE custom slicing, columns are: {list(df.columns)} (rows={df.shape[0]})")
    df = df.iloc[2:].reset_index(drop=True)
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True)
    logger.info(f"AFTER making row 0 a header, columns are: {list(df.columns)} (rows={df.shape[0]})")
    logger.info(f"Dropping the last {drop_cols} columns...")
    if drop_cols > 0:
        df = df.iloc[:, :-drop_cols]
    df = df.fillna("NONE")
    logger.info(f"AFTER dropping last {drop_cols} cols and fillna, columns: {list(df.columns)} (rows={df.shape[0]})")
    return df

# ----------------------------------------------------------------
# Run the Flask Apps
# ----------------------------------------------------------------
# if __name__ == "__main__":
#     try:
#         ensure_model_directory()
#         logger.info("Loading model pipeline once at startup...")
#         asyncio.run(async_load_model_pipeline())
#         app.run(host="0.0.0.0", port=5000)
#     except Exception as e:
#         logger.error(f"Application failed to start: {e}\n{traceback.format_exc()}")
#         sys.exit(1)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# import joblib
# from flask import Flask, request, send_file, render_template, redirect, url_for, flash, jsonify
# import pandas as pd
# import logging
# import traceback
# import sys
# import hashlib
# import asyncio  # For asynchronous operations
# import aiofiles  # For asynchronous file operations

# from azure.storage.blob.aio import BlobServiceClient  # Asynchronous BlobServiceClient
# from azure.core.exceptions import AzureError
# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# # This is the same class you used when training
# from sequential_imputer import SequentialImputer

# # ----------------------------------------------------------------
# # Configure Logging
# # ----------------------------------------------------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ----------------------------------------------------------------
# # Create the Flask app
# # ----------------------------------------------------------------
# app = Flask(__name__)

# # ----------------------------------------------------------------
# # Log the SECRET_KEY from environment
# # ----------------------------------------------------------------
# logger.info("Checking environment for SECRET_KEY...")
# secret_from_env = os.environ.get("SECRET_KEY")
# logger.info(f"ENV SECRET_KEY: {secret_from_env}")

# # If environment variable is missing, we fall back to 'hello'
# app.secret_key = secret_from_env or "hello"
# logger.info(f"app.secret_key has been set to: {app.secret_key}")

# # ----------------------------------------------------------------
# # Global Paths
# # ----------------------------------------------------------------
# # HOME_DIR = os.environ.get('HOME', '/home')  # On Azure App Service, /home is persistent
# HOME_DIR = os.environ.get('ROOT', '/root')  # On Azure App Service, /root is persistent
# MODEL_DIR = os.path.join(HOME_DIR, 'models')
# MODEL_PATH = os.path.join(MODEL_DIR, 'Tam_eheatingV5.pkl')
# CHECKSUM_PATH = os.path.join(MODEL_DIR, 'Tam_eheatingV5.pkl.sha256')

# # ----------------------------------------------------------------
# # Lock and Global Model Pipeline
# # ----------------------------------------------------------------
# model_lock = asyncio.Lock()
# model_pipeline = None  # will store the loaded SequentialImputer

# # ----------------------------------------------------------------
# # Ensure Model Directory
# # ----------------------------------------------------------------
# def ensure_model_directory():
#     try:
#         os.makedirs(MODEL_DIR, exist_ok=True)
#         logger.info(f"Model directory is ready at '{MODEL_DIR}'.")
#     except Exception as e:
#         logger.error(f"Error ensuring model directory exists: {e}\n{traceback.format_exc()}")
#         raise

# # ----------------------------------------------------------------
# # Checksum Utilities
# # ----------------------------------------------------------------
# def compute_sha256(file_path):
#     sha256_hash = hashlib.sha256()
#     try:
#         with open(file_path, "rb") as f:
#             for byte_block in iter(lambda: f.read(4096), b""):
#                 sha256_hash.update(byte_block)
#         return sha256_hash.hexdigest()
#     except Exception as e:
#         logger.error(f"Error computing SHA-256 checksum for '{file_path}': {e}\n{traceback.format_exc()}")
#         raise

# def verify_checksum(model_path, checksum_path):
#     try:
#         if not os.path.exists(checksum_path):
#             raise FileNotFoundError(f"Checksum file '{checksum_path}' not found.")

#         with open(checksum_path, 'r') as f:
#             expected_checksum = f.read().split()[0].strip()

#         actual_checksum = compute_sha256(model_path)

#         logger.info(f"Expected Checksum (SHA-256): {expected_checksum}")
#         logger.info(f"Actual Checksum  (SHA-256): {actual_checksum}")

#         if actual_checksum.lower() != expected_checksum.lower():
#             raise ValueError(
#                 f"Checksum mismatch for '{model_path}'. "
#                 f"Expected {expected_checksum}, got {actual_checksum}."
#             )

#         logger.info(f"Checksum verification passed for '{model_path}'.")
#     except Exception as e:
#         logger.error(f"Error verifying checksum for '{model_path}': {e}\n{traceback.format_exc()}")
#         raise

# # ----------------------------------------------------------------
# # Async Download with Retry
# # ----------------------------------------------------------------
# @retry(
#     reraise=True,
#     stop=stop_after_attempt(5),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type((EnvironmentError, FileNotFoundError, AzureError))
# )
# async def async_download_blob(container_name, blob_name, download_file_path):
#     try:
#         os.makedirs(os.path.dirname(download_file_path), exist_ok=True)

#         connect_str = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
#         if not connect_str:
#             raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING is not set.")

#         blob_service_client = BlobServiceClient.from_connection_string(connect_str)
#         blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

#         exists = await blob_client.exists()
#         if not exists:
#             raise FileNotFoundError(f"Blob '{blob_name}' not found in container '{container_name}'.")

#         logger.info(f"Starting download of blob '{blob_name}' from container '{container_name}'.")
#         async with aiofiles.open(download_file_path, "wb") as download_file:
#             download_stream = await blob_client.download_blob()
#             async for chunk in download_stream.chunks():
#                 await download_file.write(chunk)

#         logger.info(f"Downloaded blob '{blob_name}' to '{download_file_path}'.")
#     except Exception as e:
#         logger.error(f"Error downloading blob '{blob_name}': {e}\n{traceback.format_exc()}")
#         raise

# # ----------------------------------------------------------------
# # Download & Verify Model
# # ----------------------------------------------------------------
# async def async_download_and_verify_model():
#     try:
#         # 1) Download the .pkl
#         await async_download_blob(
#             container_name='models',
#             blob_name='Tam_eheatingV5.pkl',
#             download_file_path=MODEL_PATH
#         )
#         # 2) Download the .sha256
#         await async_download_blob(
#             container_name='models',
#             blob_name='Tam_eheatingV5.pkl.sha256',
#             download_file_path=CHECKSUM_PATH
#         )
#         # 3) Verify
#         verify_checksum(MODEL_PATH, CHECKSUM_PATH)
#     except Exception as e:
#         logger.error(f"Error in async_download_and_verify_model: {e}\n{traceback.format_exc()}")
#         raise


# def sync_download_and_verify_model():
#     """
#     Synchronously download and verify the model from Azure Blob Storage.
#     """
#     import asyncio
#     try:
#         asyncio.run(async_download_and_verify_model())
#     except Exception as e:
#         logger.error(f"Failed to download/verify model at startup: {e}\n{traceback.format_exc()}")
#         raise


# def sync_load_model_pipeline():
#     """
#     Synchronously load the model pipeline into memory.
#     """
#     global model_pipeline
#     try:
#         ensure_model_directory()
#         if not os.path.exists(MODEL_PATH):
#             logger.error(f"Model file not found at {MODEL_PATH}")
#             raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
#         model_pipeline = joblib.load(MODEL_PATH)
#         logger.info("Model pipeline loaded into memory.")
#     except Exception as e:
#         logger.error(f"Failed to load model pipeline: {e}\n{traceback.format_exc()}")
#         raise
# # ----------------------------------------------------------------
# # Load the Trained Imputer at Startup
# # ----------------------------------------------------------------
# async def async_load_model_pipeline():
#     global model_pipeline
#     if model_pipeline is not None:
#         logger.info("Model pipeline already loaded. Skipping download.")
#         return

#     async with model_lock:
#         if model_pipeline is not None:
#             logger.info("Model pipeline already loaded inside lock. Skipping download.")
#             return
#         try:
#             ensure_model_directory()
#             logger.info("Starting asynchronous model download and verification...")
#             await async_download_and_verify_model()

#             logger.info(f"Loading model from '{MODEL_PATH}'...")
#             loop = asyncio.get_event_loop()
#             model_pipeline = await loop.run_in_executor(None, joblib.load, MODEL_PATH)
#             logger.info("Model pipeline loaded successfully.")
#         except Exception as e:
#             logger.error(f"Error loading model: {e}\n{traceback.format_exc()}")
#             raise

# # ----------------------------------------------------------------
# # Flask Routes
# # ----------------------------------------------------------------
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/health')
# def health():
#     """
#     Health check endpoint for Azure App Service.
#     """
#     return jsonify({"status": "OK"}), 200

# @app.route('/predict', methods=['POST'])
# def predict():
#     """
#     This endpoint:
#       1) Receives an Excel file
#       2) Reads it into a DataFrame (using openpyxl engine)
#       3) Preprocesses (so that columns align with the trained imputer)
#       4) Calls model_pipeline.transform(...) to fill missing/impute columns
#       5) Returns the imputed DataFrame as 'predictions.xlsx'
#     """
#     if 'file' not in request.files:
#         logger.warning("No file part in the request.")
#         flash("No file part in the request.", "danger")
#         return redirect(url_for('home'))

#     uploaded_file = request.files['file']
#     if uploaded_file.filename == '':
#         logger.warning("No selected file.")
#         flash("No selected file.", "danger")
#         return redirect(url_for('home'))

#     if not allowed_file(uploaded_file.filename):
#         logger.warning("Unsupported file type.")
#         flash("Unsupported file type. Please upload an Excel file (.xls or .xlsx).", "danger")
#         return redirect(url_for('home'))

#     try:
#         # 1) Read Excel file into a DataFrame
#         input_data = pd.read_excel(uploaded_file, engine="openpyxl")
#         logger.info("Received Excel file for imputation.")
#         logger.info(f"RAW columns from uploaded Excel: {list(input_data.columns)}")

#         # 2) Preprocess input_data so columns match exactly what the imputer expects
#         input_data = preprocess_input_data(input_data)
#         logger.info("Preprocessed input data to match training.")
#         logger.info(f"After preprocessing, columns are: {list(input_data.columns)}")
# # ----------------------------------------- NEW CODE -----------------------------------------
#         try:
#             ensure_model_directory()
#             logger.info("Downloading and verifying model at startup...")
#             sync_download_and_verify_model()
#             logger.info("Loading model pipeline at startup...")
#             sync_load_model_pipeline()
#         except Exception as e:
#             logger.error(f"Application failed to start: {e}\n{traceback.format_exc()}")
# # ----------------------------------------- NEW CODE -----------------------------------------

#         # 3) Check if model pipeline is loaded
#         if model_pipeline is None:
#             logger.warning("Model pipeline not loaded in memory. Aborting.")
#             flash("Model pipeline not loaded in memory.", "danger")
#             return redirect(url_for('home'))

#         # 4) Transform the data to fill missing columns
#         logger.info("Imputing data with model_pipeline.transform(...)")
#         imputed_data = model_pipeline.transform(input_data)
#         logger.info("Data imputation completed successfully.")

#         # 5) Save the imputed DataFrame as 'predictions.xlsx'
#         output_file = 'predictions.xlsx'
#         imputed_data.to_excel(output_file, index=False)

#         logger.info(f"Imputed data saved to '{output_file}'.")

#         # 6) Return the Excel file to the user
#         return send_file(
#             output_file,
#             as_attachment=True,
#             mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
#         )

#     except Exception as e:
#         error_message = f"Error during imputation: {e}\n{traceback.format_exc()}"
#         logger.error(error_message)
#         flash(f"An error occurred during imputation: {str(e)}", "danger")
#         return redirect(url_for('home'))

# # ----------------------------------------------------------------
# # Helper Functions
# # ----------------------------------------------------------------
# def allowed_file(filename):
#     """
#     Checks if the uploaded file is an Excel file (.xls or .xlsx).
#     """
#     if '.' not in filename:
#         return False
#     ext = filename.rsplit('.', 1)[1].lower()
#     return ext in ["xls", "xlsx"]

# def preprocess_input_data(df):
#     """
#     IMPORTANT:
#     Replicate the same cleaning steps you did locally before training.
#     For example:
#        df = df.iloc[2:].reset_index(drop=True)
#        df.columns = df.iloc[0]
#        df = df.drop(df.index[0]).reset_index(drop=True)
#        df = df.iloc[:, :-17]
#        df = df.fillna("NONE")

#     We'll do exactly that here to ensure your columns
#     (like 'Cabin Heat Technology') exist and match
#     the names your imputer was trained on.
#     """
#     logger.info(f"BEFORE custom slicing, columns are: {list(df.columns)} (rows={df.shape[0]})")

#     # Skip the first two rows
#     df = df.iloc[2:].reset_index(drop=True)

#     # Let the next row become the column headers
#     df.columns = df.iloc[0]
#     df = df.drop(df.index[0]).reset_index(drop=True)

#     logger.info(f"AFTER making row 0 a header, columns are: {list(df.columns)} (rows={df.shape[0]})")

#     # Drop the last 17 columns
#     logger.info("Dropping the last 17 columns...")
#     df = df.iloc[:, :-17]

#     # Fill missing values with "NONE"
#     df = df.fillna("NONE")

#     logger.info(f"AFTER dropping last 17 cols and fillna, columns: {list(df.columns)} (rows={df.shape[0]})")

#     return df

# # ----------------------------------------------------------------
# # Run the Flask Apps
# # ----------------------------------------------------------------
# # if __name__ == "__main__":
# #     try:
# #         ensure_model_directory()
# #         print(f"Model directory is ready at '{MODEL_DIR}'.")
# #         logger.info("Loading model pipeline once at startup...")
# #         asyncio.run(async_load_model_pipeline())
# #         app.run(host="0.0.0.0", port=5000)
# #     except Exception as e:
# #         logger.error(f"Application failed to start: {e}\n{traceback.format_exc()}")
# #         sys.exit(1)

# # try:
# #     ensure_model_directory()
# #     logger.info("Downloading and verifying model at startup...")
# #     sync_download_and_verify_model()
# #     logger.info("Loading model pipeline at startup...")
# #     sync_load_model_pipeline()
# # except Exception as e:
# #     logger.error(f"Application failed to start: {e}\n{traceback.format_exc()}")
#     # Optionally: raise



# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)