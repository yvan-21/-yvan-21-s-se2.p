import copy
import os
import uuid
import json
import csv
import shutil
import functools
import io
import hashlib
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file, send_from_directory
from werkzeug.utils import secure_filename
import logging
import numpy as np
from PIL import Image, ImageOps
import base64
from io import BytesIO
import glob
import random
import threading
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import mysql.connector
from unknown_detection import ValidationCalibratedDetector

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras
from keras import layers, models
import cv2
from skimage.feature import graycomatrix, graycoprops
from background_removal import remove_background as extract_foreground

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel('ERROR')

app = Flask(__name__)

# =========================================================
# CONFIGURATION
# =========================================================
APP_NAME = "Orchid Identifier"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
SCAN_RECORDS_FILE = os.path.join(BASE_DIR, "scan_records.json")
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "orchd_sep2_system"),
}
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.config["APP_NAME"] = APP_NAME
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["DATASET_FOLDER"] = DATASET_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_database_connection():
    if not MYSQL_CONFIG["password"]:
        raise RuntimeError("MYSQL_PASSWORD is not set in this process environment")

    connection = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS app_documents (
            document_key VARCHAR(100) PRIMARY KEY,
            payload JSON NOT NULL,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    cursor.close()
    return connection


def get_document_file_path(document_key, legacy_file=None):
    if legacy_file:
        return legacy_file
    if document_key == "training_metrics":
        return os.path.join(BASE_DIR, "models", "training_metrics.json")
    if document_key == "scan_records":
        return SCAN_RECORDS_FILE
    if document_key == "users":
        return os.path.join(BASE_DIR, "users.json")
    if document_key == "recommendations":
        return os.path.join(BASE_DIR, "recommendations.json")
    if document_key == "settings":
        return os.path.join(BASE_DIR, "settings.json")
    return os.path.join(BASE_DIR, f"{document_key}.json")


def load_local_app_document(document_key, legacy_file=None, default=None, create_default=True):
    file_path = get_document_file_path(document_key, legacy_file)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as data_file:
            return json.load(data_file)
    if default is not None and create_default:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as data_file:
            json.dump(default, data_file)
    return default


def save_local_app_document(document_key, payload):
    file_path = get_document_file_path(document_key)
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as data_file:
        json.dump(payload, data_file)


def load_app_document(document_key, legacy_file=None, default=None):
    if not MYSQL_CONFIG["password"]:
        return load_local_app_document(document_key, legacy_file, default)

    try:
        connection = get_database_connection()
    except mysql.connector.Error:
        logger.exception(
            "Database unavailable while loading '%s'; falling back to local JSON",
            document_key
        )
        return load_local_app_document(
            document_key,
            legacy_file,
            default,
            create_default=False
        )
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT payload FROM app_documents WHERE document_key = %s",
            (document_key,)
        )
        row = cursor.fetchone()
        if row:
            payload = row[0]
            return json.loads(payload) if isinstance(payload, (str, bytes)) else payload

        payload = default
        if legacy_file and os.path.exists(legacy_file):
            with open(legacy_file, "r", encoding="utf-8") as legacy_data:
                payload = json.load(legacy_data)
        cursor.execute(
            "INSERT IGNORE INTO app_documents (document_key, payload) VALUES (%s, %s)",
            (document_key, json.dumps(payload))
        )
        connection.commit()
        return payload
    except mysql.connector.Error:
        logger.exception(
            "Database query failed while loading '%s'; falling back to local JSON",
            document_key
        )
        return load_local_app_document(
            document_key,
            legacy_file,
            default,
            create_default=False
        )
    finally:
        connection.close()


def save_app_document(document_key, payload):
    if not MYSQL_CONFIG["password"]:
        save_local_app_document(document_key, payload)
        return

    try:
        connection = get_database_connection()
    except mysql.connector.Error:
        logger.exception(
            "Database unavailable while saving '%s'; saving to local JSON",
            document_key
        )
        save_local_app_document(document_key, payload)
        return
    try:
        cursor = connection.cursor()
        cursor.execute(
            """INSERT INTO app_documents (document_key, payload)
               VALUES (%s, %s)
               ON DUPLICATE KEY UPDATE payload = VALUES(payload)""",
            (document_key, json.dumps(payload))
        )
        connection.commit()
    except mysql.connector.Error:
        logger.exception(
            "Database write failed for '%s'; saving to local JSON",
            document_key
        )
        connection.rollback()
        save_local_app_document(document_key, payload)
    finally:
        connection.close()

# =========================================================
# AUTHENTICATION
# =========================================================
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# =========================================================
# SCAN RECORDS MANAGEMENT
# =========================================================
def load_scan_records():
    """Load scan records from MySQL, importing the legacy JSON file once."""
    try:
        return load_app_document("scan_records", SCAN_RECORDS_FILE, [])
    except Exception as e:
        logger.error(f"Failed to load scan records: {e}")
        return []

def save_scan_record(record):
    """Save a single scan record to JSON file"""
    try:
        records = load_scan_records()
        record['timestamp'] = datetime.now().isoformat()
        record['id'] = str(uuid.uuid4())
        records.insert(0, record)  # Add to beginning

        # Keep only last 1000 records to prevent file from growing too large
        if len(records) > 1000:
            records = records[:1000]

        save_app_document("scan_records", records)
        return True
    except Exception as e:
        logger.error(f"Failed to save scan record: {e}")
        return False

def get_scan_statistics():
    """Calculate scan statistics"""
    records = load_scan_records()
    now = datetime.now()
    today = now.date()
    week_ago = now - datetime.timedelta(days=7)

    total_scans = len(records)
    today_scans = len([r for r in records if datetime.fromisoformat(r['timestamp']).date() == today])
    week_scans = len([r for r in records if datetime.fromisoformat(r['timestamp']) >= week_ago])

    return {
        "total_scans": total_scans,
        "today_scans": today_scans,
        "week_scans": week_scans
    }

# =========================================================
# TRAINING PROGRESS TRACKING
# =========================================================
training_progress = {
    "status": "idle",
    "percentage": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "current_stage": "",
    "message": ""
}
training_lock = threading.Lock()

class ProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to track training progress"""
    def __init__(self, total_epochs, stage_name, start_percentage, end_percentage):
        super().__init__()
        self.total_epochs = total_epochs
        self.stage_name = stage_name
        self.start_percentage = start_percentage
        self.end_percentage = end_percentage

    def on_train_batch_end(self, batch, logs=None):
        global training_progress
        steps = self.params.get("steps") or 1
        epoch_progress = (self.current_epoch + min(batch + 1, steps) / steps) / self.total_epochs
        training_progress["percentage"] = self.start_percentage + (
            self.end_percentage - self.start_percentage
        ) * epoch_progress
        training_progress["message"] = (
            f"{self.stage_name} - Epoch {self.current_epoch + 1}/{self.total_epochs}, "
            f"batch {batch + 1}/{steps}"
        )

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        global training_progress
        training_progress["current_epoch"] = epoch + 1
        training_progress["total_epochs"] = self.total_epochs
        training_progress["current_stage"] = self.stage_name
        training_progress["message"] = f"{self.stage_name} - Epoch {epoch + 1}/{self.total_epochs}"
        logger.info(f"Progress: {self.stage_name} - Epoch {epoch + 1}/{self.total_epochs}")

# =========================================================
# FILE VALIDATION
# =========================================================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def verify_image_file(file_storage):
    try:
        img = Image.open(file_storage)
        img.verify()
        file_storage.seek(0)
        return True
    except Exception:
        return False

# =========================================================
# DATASET FUNCTIONS
# =========================================================
def get_dataset_samples():
    """Get sample images from dataset folder"""
    samples = []
    species_list = ['Phalaenopsis', 'Cattleya', 'Dendrobium', 'Vanda']
    
    for species in species_list:
        species_path = os.path.join(DATASET_DIR, species)
        if os.path.exists(species_path):
            # Get healthy samples
            healthy_path = os.path.join(species_path, 'healthy')
            if os.path.exists(healthy_path):
                healthy_images = glob.glob(os.path.join(healthy_path, '*.*'))
                healthy_images = [f for f in healthy_images if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
                for img_path in healthy_images[:2]:
                    filename = os.path.basename(img_path)
                    samples.append({
                        'species': species,
                        'health': 'healthy',
                        'image_url': f'/static/dataset/{species}/healthy/{filename}',
                        'filename': filename,
                        'is_placeholder': False
                    })
            
            # Get not healthy samples
            not_healthy_path = os.path.join(species_path, 'not_healthy')
            if os.path.exists(not_healthy_path):
                not_healthy_images = glob.glob(os.path.join(not_healthy_path, '*.*'))
                not_healthy_images = [f for f in not_healthy_images if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
                for img_path in not_healthy_images[:2]:
                    filename = os.path.basename(img_path)
                    samples.append({
                        'species': species,
                        'health': 'not_healthy',
                        'image_url': f'/static/dataset/{species}/not_healthy/{filename}',
                        'filename': filename,
                        'is_placeholder': False
                    })
    
    if not samples:
        for species in species_list:
            samples.append({
                'species': species,
                'health': 'healthy',
                'image_url': None,
                'is_placeholder': True,
                'filename': 'No images found'
            })
            samples.append({
                'species': species,
                'health': 'not_healthy',
                'image_url': None,
                'is_placeholder': True,
                'filename': 'No images found'
            })
    
    random.shuffle(samples)
    return samples[:8]

def get_dataset_stats():
    """Get statistics about the dataset"""
    stats = {
        'total_images': 0,
        'species_count': {},
        'health_count': {'healthy': 0, 'not_healthy': 0}
    }
    
    species_list = ['Phalaenopsis', 'Cattleya', 'Dendrobium', 'Vanda']
    
    for species in species_list:
        stats['species_count'][species] = {'healthy': 0, 'not_healthy': 0, 'total': 0}
        
        for health in ['healthy', 'not healthy', 'not_healthy']:
            path = os.path.join(DATASET_DIR, species, health)
            if os.path.exists(path):
                images = glob.glob(os.path.join(path, '*.*'))
                images = [f for f in images if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
                count = len(images)
                
                # Normalize health status key
                health_key = 'not_healthy' if health in ['not healthy', 'not_healthy'] else 'healthy'
                stats['species_count'][species][health_key] = count
                stats['species_count'][species]['total'] += count
                stats['health_count'][health_key] += count
                stats['total_images'] += count
    
    return stats

def get_dataset_image_paths():
    """Return dataset image paths relative to DATASET_DIR with normalized separators."""
    image_paths = []
    for species in orchid_ai.species_classes:
        for health in ('healthy', 'not healthy', 'not_healthy'):
            folder = os.path.join(DATASET_DIR, species, health)
            if not os.path.isdir(folder):
                continue
            for full_path in glob.glob(os.path.join(folder, '*.*')):
                if full_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                    image_paths.append(
                        os.path.relpath(full_path, DATASET_DIR).replace(os.sep, '/')
                    )
    return sorted(image_paths)


def get_dataset_upload_registry_path():
    return os.path.join(BASE_DIR, "models", "dataset_uploads.json")


def get_new_dataset_upload_paths():
    registry_path = get_dataset_upload_registry_path()
    if not os.path.exists(registry_path):
        return set()
    try:
        with open(registry_path, "r", encoding="utf-8") as registry_file:
            registry = json.load(registry_file)
        paths = registry.get("new_upload_paths", [])
        if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
            raise ValueError("new_upload_paths must be a list of image paths")
        return {path.replace("\\", "/").removeprefix("./") for path in paths}
    except (OSError, json.JSONDecodeError, AttributeError, ValueError) as error:
        raise ValueError(f"Could not read uploaded dataset image registry: {error}") from error


def get_training_image_paths(scope):
    if scope not in {"old", "new", "all"}:
        raise ValueError("Training scope must be old, new, or all.")
    all_paths = get_dataset_image_paths()
    new_paths = get_new_dataset_upload_paths().intersection(all_paths)
    if scope == "new":
        return sorted(new_paths)
    if scope == "old":
        return sorted(set(all_paths) - new_paths)
    return all_paths


def mark_dataset_image_as_new_upload(image_path):
    relative_path = image_path.replace("\\", "/").removeprefix("./")
    registry_path = get_dataset_upload_registry_path()
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    new_paths = get_new_dataset_upload_paths()
    new_paths.add(relative_path)
    temporary_path = f"{registry_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as registry_file:
        json.dump({"new_upload_paths": sorted(new_paths)}, registry_file, indent=2)
        registry_file.flush()
        os.fsync(registry_file.fileno())
    os.replace(temporary_path, registry_path)


def register_uploaded_image_group(image_path):
    image_paths = get_dataset_image_paths()
    manifest_path = get_image_group_manifest_path()
    if not os.path.exists(manifest_path):
        create_image_group_manifest_template()
    with open(manifest_path, newline="", encoding="utf-8-sig") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required_columns = {"image_path", "plant_id", "session_id"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "The image group CSV must have image_path, plant_id, and session_id columns."
            )
        rows = list(reader)

    normalized_path = image_path.replace("\\", "/").removeprefix("./")
    current_paths = set(image_paths)
    normalized_rows = []
    for row in rows:
        row_path = row["image_path"]
        if row_path in current_paths:
            normalized_rows.append(row)
    if len({row["image_path"] for row in normalized_rows}) != len(normalized_rows):
        raise ValueError("The image group CSV contains duplicate image paths.")
    rows = normalized_rows
    known_paths = {row["image_path"] for row in rows}
    rows.extend(
        {"image_path": path, "plant_id": "", "session_id": ""}
        for path in sorted(current_paths - known_paths)
    )

    found = False
    for row in rows:
        row_path = (row.get("image_path") or "").strip().replace("\\", "/").removeprefix("./")
        row["image_path"] = row_path
        row["plant_id"] = (row.get("plant_id") or "").strip()
        row["session_id"] = (row.get("session_id") or "").strip()
        if row_path == normalized_path:
            row["plant_id"] = f"upload_{uuid.uuid4().hex}"
            row["session_id"] = ""
            found = True
    if not found:
        rows.append({
            "image_path": normalized_path,
            "plant_id": f"upload_{uuid.uuid4().hex}",
            "session_id": "",
        })
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["image_path", "plant_id", "session_id"],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["image_path"]))


def get_dataset_fingerprint(image_paths, image_group_ids=None):
    """Fingerprint current image paths and file metadata for metric freshness checks."""
    digest = hashlib.sha256()
    for image_path in sorted(image_paths):
        full_path = os.path.join(DATASET_DIR, image_path.replace("/", os.sep))
        stat = os.stat(full_path)
        digest.update(image_path.encode("utf-8"))
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))
        if image_group_ids is not None:
            digest.update(image_group_ids[image_path].encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def has_generated_placeholder_group_ids(image_paths, rows_by_path):
    """Identify legacy per-image IDs generated instead of real plant/session annotations."""
    if not image_paths:
        return False
    assigned_ids = set()
    for image_path in image_paths:
        plant_id = rows_by_path[image_path]["plant_id"]
        session_id = rows_by_path[image_path]["session_id"]
        if not plant_id.startswith("plant_") or not session_id.startswith("session_"):
            return False
        plant_number = plant_id.removeprefix("plant_")
        session_number = session_id.removeprefix("session_")
        if not plant_number.isdigit() or plant_number != session_number:
            return False
        assigned_ids.add(plant_number)
    return len(assigned_ids) == len(image_paths)


def get_image_group_manifest_path():
    return os.path.join(DATASET_DIR, "image_groups.csv")

def create_image_group_manifest_template():
    """Create a fill-in CSV template without overwriting existing group annotations."""
    manifest_path = get_image_group_manifest_path()
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w", newline="", encoding="utf-8-sig") as manifest_file:
            writer = csv.DictWriter(
                manifest_file,
                fieldnames=["image_path", "plant_id", "session_id"]
            )
            writer.writeheader()
            for image_path in get_dataset_image_paths():
                writer.writerow({
                    "image_path": image_path,
                    "plant_id": "",
                    "session_id": ""
                })
    return manifest_path


def repair_missing_group_manifest_ids(image_paths):
    """Refresh manifest paths and assign unique IDs to images without annotations."""
    manifest_path = get_image_group_manifest_path()
    rows_by_path = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, newline="", encoding="utf-8-sig") as manifest_file:
                reader = csv.DictReader(manifest_file)
                required_columns = {"image_path", "plant_id", "session_id"}
                if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                    raise ValueError(
                        "The group CSV must have image_path, plant_id, and session_id columns."
                    )
                for row in reader:
                    image_path = (row.get("image_path") or "").strip().replace("\\", "/")
                    image_path = image_path.removeprefix("./")
                    if not image_path:
                        continue
                    if image_path in rows_by_path:
                        raise ValueError(
                            f"The group CSV contains duplicate image paths: {image_path}"
                        )
                    rows_by_path[image_path] = {
                        "plant_id": (row.get("plant_id") or "").strip(),
                        "session_id": (row.get("session_id") or "").strip(),
                    }
        except (OSError, csv.Error, UnicodeError) as error:
            logger.exception("Could not repair the group CSV")
            raise ValueError(f"Could not read the existing group CSV: {error}") from error

    changed = False
    sorted_paths = sorted(set(image_paths))
    for image_path in sorted_paths:
        row = rows_by_path.get(image_path, {"plant_id": "", "session_id": ""})
        plant_id = (row.get("plant_id") or "").strip()
        session_id = (row.get("session_id") or "").strip()

        if image_path not in rows_by_path:
            changed = True

        if not plant_id and not session_id:
            plant_id = f"upload_{uuid.uuid4().hex}"
            changed = True

        rows_by_path[image_path] = {"plant_id": plant_id, "session_id": session_id}

    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=["image_path", "plant_id", "session_id"])
        writer.writeheader()
        for image_path in sorted_paths:
            row = rows_by_path.get(image_path, {"plant_id": "", "session_id": ""})
            writer.writerow({
                "image_path": image_path,
                "plant_id": row.get("plant_id", ""),
                "session_id": row.get("session_id", "")
            })

    return changed


def render_image_group_manifest_template():
    """Build a current-path template while preserving annotations for matching images."""
    current_paths = get_dataset_image_paths()
    current_path_set = set(current_paths)
    annotations = {}
    manifest_path = get_image_group_manifest_path()
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, newline="", encoding="utf-8-sig") as manifest_file:
                reader = csv.DictReader(manifest_file)
                required_columns = {"image_path", "plant_id", "session_id"}
                if reader.fieldnames and required_columns.issubset(reader.fieldnames):
                    for row in reader:
                        image_path = (row.get("image_path") or "").strip().replace("\\", "/")
                        image_path = image_path.removeprefix("./")
                        if image_path in current_path_set:
                            annotations[image_path] = {
                                "plant_id": (row.get("plant_id") or "").strip(),
                                "session_id": (row.get("session_id") or "").strip()
                            }
        except (OSError, csv.Error, UnicodeError) as error:
            logger.exception("Could not preserve existing group CSV annotations in template")
            raise ValueError(f"Could not read the existing group CSV: {error}") from error

    if (
        len(annotations) == len(current_paths)
        and has_generated_placeholder_group_ids(current_paths, annotations)
    ):
        annotations = {}

    template = io.StringIO(newline="")
    writer = csv.DictWriter(
        template,
        fieldnames=["image_path", "plant_id", "session_id"]
    )
    writer.writeheader()
    for image_path in current_paths:
        writer.writerow({
            "image_path": image_path,
            **annotations.get(image_path, {"plant_id": "", "session_id": ""})
        })
    return io.BytesIO(template.getvalue().encode("utf-8-sig"))

def load_image_group_assignments(image_paths, allow_extra_paths=False):
    """Load and validate explicit plant/session IDs for every dataset image."""
    manifest_path = get_image_group_manifest_path()
    if not os.path.exists(manifest_path):
        return None, f"Group manifest is missing: {manifest_path}"

    rows_by_path = {}
    try:
        with open(manifest_path, newline="", encoding="utf-8-sig") as manifest_file:
            reader = csv.DictReader(manifest_file)
            required_columns = {"image_path", "plant_id", "session_id"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                return None, (
                    "The group CSV must have image_path, plant_id, and session_id columns."
                )
            for row_number, row in enumerate(reader, start=2):
                image_path = (row.get("image_path") or "").strip().replace("\\", "/")
                image_path = image_path.removeprefix("./")
                if not image_path:
                    return None, f"Group CSV row {row_number} has an empty image_path."
                if image_path in rows_by_path:
                    return None, f"Group CSV contains duplicate image_path: {image_path}"
                rows_by_path[image_path] = {
                    "plant_id": (row.get("plant_id") or "").strip(),
                    "session_id": (row.get("session_id") or "").strip()
                }
    except (OSError, csv.Error, UnicodeError) as error:
        logger.exception("Failed to read image group manifest")
        return None, f"Could not read the image group CSV: {error}"

    current_paths = set(image_paths)
    manifest_paths = set(rows_by_path)
    missing_paths = sorted(current_paths - manifest_paths)
    actual_paths = set(get_dataset_image_paths()) if allow_extra_paths else current_paths
    removed_paths = sorted(manifest_paths - actual_paths)
    if missing_paths:
        return None, (
            f"Group CSV is missing {len(missing_paths)} dataset image row(s), "
            f"including {missing_paths[0]}."
        )
    if removed_paths:
        return None, (
            f"Group CSV contains {len(removed_paths)} image path(s) no longer in the dataset, "
            f"including {removed_paths[0]}."
        )

    image_owners = {}
    duplicate_image_pairs = []
    for index, image_path in enumerate(image_paths):
        full_path = os.path.join(DATASET_DIR, image_path.replace("/", os.sep))
        if not os.path.isfile(full_path):
            continue
        with open(full_path, "rb") as image_file:
            image_digest = hashlib.sha256(image_file.read()).digest()
        path_parts = image_path.split("/")
        label = (
            path_parts[0],
            "healthy" if path_parts[1] == "healthy" else "not_healthy"
        )
        if image_digest in image_owners:
            owner_index, owner_label, owner_path = image_owners[image_digest]
            if owner_label != label:
                return None, (
                    f"Identical image content has conflicting dataset labels: "
                    f"{owner_path} and {image_path}. Correct or remove the duplicate "
                    "before training."
                )
            duplicate_image_pairs.append((index, owner_index))
        else:
            image_owners[image_digest] = (index, label, image_path)

    if has_generated_placeholder_group_ids(image_paths, rows_by_path):
        return None, (
            "The group CSV contains automatically generated per-image IDs, not verified "
            "plant/session annotations. Replace them with the real plant_id and/or session_id "
            "for each photo before training."
        )

    parent = list(range(len(image_paths)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first, second in duplicate_image_pairs:
        union(first, second)

    group_owner = {}
    path_to_index = {image_path: index for index, image_path in enumerate(image_paths)}
    for image_path in image_paths:
        row = rows_by_path[image_path]
        plant_id = row["plant_id"]
        session_id = row["session_id"]
        if not plant_id and not session_id:
            return None, (
                f"Add a plant_id or session_id for every image; no group ID is set for {image_path}."
            )

        for group_key in (
            f"plant:{plant_id}" if plant_id else None,
            f"session:{session_id}" if session_id else None
        ):
            if group_key is None:
                continue
            index = path_to_index[image_path]
            if group_key in group_owner:
                union(index, group_owner[group_key])
            else:
                group_owner[group_key] = index

    return {
        image_path: f"group-{find(index)}"
        for index, image_path in enumerate(image_paths)
    }, None

def get_dataset_labels_for_paths(image_paths):
    species_index = {
        species: index
        for index, species in enumerate(orchid_ai.species_classes)
    }
    species_labels = []
    health_labels = []
    for image_path in image_paths:
        path_parts = image_path.split("/")
        if len(path_parts) != 3 or path_parts[0] not in species_index:
            raise ValueError(f"Unexpected dataset image path structure: {image_path}")
        if path_parts[1] not in {"healthy", "not healthy", "not_healthy"}:
            raise ValueError(f"Unexpected health label in dataset path: {image_path}")
        species_labels.append(species_index[path_parts[0]])
        health_labels.append(0 if path_parts[1] == "healthy" else 1)
    species_labels = np.asarray(species_labels)
    health_labels = np.asarray(health_labels)
    if not np.array_equal(np.unique(species_labels), np.arange(len(species_index))):
        raise ValueError("Every configured species must have at least one dataset image.")
    if not np.array_equal(np.unique(health_labels), np.array([0, 1])):
        raise ValueError("The dataset must include both healthy and unhealthy images.")
    return species_labels, health_labels

def make_grouped_train_validation_test_split(labels, groups, target_name):
    """Create deterministic, stratified train/validation/test partitions with disjoint groups."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    classes = np.unique(labels)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 8:
        raise ValueError(
            f"{target_name} needs at least 8 distinct plant/session groups for a "
            "three-way group-safe split; add more independently grouped images."
        )

    best_split = None
    best_score = float("inf")
    for outer_seed in range(42, 52):
        outer_splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=outer_seed
        )
        for remaining_indices, test_indices in outer_splitter.split(
            np.zeros(len(labels)),
            labels,
            groups
        ):
            if not np.array_equal(np.unique(labels[test_indices]), classes):
                continue
            if not np.array_equal(np.unique(labels[remaining_indices]), classes):
                continue

            inner_labels = labels[remaining_indices]
            inner_groups = groups[remaining_indices]
            if len(np.unique(inner_groups)) < 5:
                continue

            for inner_seed in range(52, 62):
                inner_splitter = StratifiedGroupKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=inner_seed
                )
                for train_local_indices, validation_local_indices in inner_splitter.split(
                    np.zeros(len(remaining_indices)),
                    inner_labels,
                    inner_groups
                ):
                    train_indices = remaining_indices[train_local_indices]
                    validation_indices = remaining_indices[validation_local_indices]
                    if not np.array_equal(np.unique(labels[train_indices]), classes):
                        continue
                    if not np.array_equal(np.unique(labels[validation_indices]), classes):
                        continue

                    train_groups = set(groups[train_indices])
                    validation_groups = set(groups[validation_indices])
                    test_groups = set(groups[test_indices])
                    if (
                        train_groups & validation_groups
                        or train_groups & test_groups
                        or validation_groups & test_groups
                    ):
                        raise RuntimeError("Group-safe split unexpectedly shares a group.")

                    sizes = (
                        len(train_indices) / len(labels),
                        len(validation_indices) / len(labels),
                        len(test_indices) / len(labels)
                    )
                    size_score = (
                        abs(sizes[0] - 0.64)
                        + abs(sizes[1] - 0.16)
                        + abs(sizes[2] - 0.20)
                    )
                    distribution_score = 0.0
                    for indices, target_fraction in (
                        (train_indices, 0.64),
                        (validation_indices, 0.16),
                        (test_indices, 0.20)
                    ):
                        for class_label in classes:
                            overall_fraction = np.mean(labels == class_label)
                            split_fraction = np.mean(labels[indices] == class_label)
                            distribution_score += target_fraction * abs(
                                split_fraction - overall_fraction
                            )
                    score = size_score + distribution_score
                    if score < best_score:
                        best_score = score
                        best_split = (train_indices, validation_indices, test_indices)

    if best_split is None:
        raise ValueError(
            f"Could not make group-safe train/validation/test splits for {target_name} "
            "with every class represented in all three partitions. Add more images from "
            "independent plants or photo sessions."
        )
    return best_split

# =========================================================
# ORCHID AI
# =========================================================
class OrchidAI:
    def __init__(self):
        self.species_model = None
        self.health_model = None
        self.species_feature_extractor = None
        self.unknown_detector = None
        self.health_base_model = None
        self._model_artifact_signature = None
        self._model_load_lock = threading.RLock()
        self.species_classes = self.get_available_species()
        self.health_classes = ['healthy', 'not_healthy']
        self.img_size = (224, 224)
        self.use_real_model = False  # Will be True after training

    def load_saved_models(self):
        """Load the newest complete trained model set when artifacts are available."""
        models_dir = os.path.join(BASE_DIR, "models")
        artifact_sets = (
            (
                "feature",
                os.path.join(models_dir, "species_feature_extractor_224.keras"),
                os.path.join(models_dir, "species_classifier_224.joblib"),
                os.path.join(models_dir, "health_classifier_224.joblib"),
            ),
            (
                "feature",
                os.path.join(models_dir, "species_feature_extractor.keras"),
                os.path.join(models_dir, "species_classifier.joblib"),
                os.path.join(models_dir, "health_classifier.joblib"),
            ),
            (
                "legacy",
                os.path.join(models_dir, "species_model.h5"),
                os.path.join(models_dir, "health_model.h5"),
            ),
        )

        selected_artifacts = next(
            (
                artifact_set
                for artifact_set in artifact_sets
                if all(os.path.isfile(path) for path in artifact_set[1:])
            ),
            None,
        )
        if selected_artifacts is None:
            return bool(
                self.use_real_model
                and self.species_model is not None
                and self.health_model is not None
            )

        artifact_stats = [
            (path, os.stat(path))
            for path in selected_artifacts[1:]
        ]
        detector_path = os.path.join(models_dir, "unknown_detection_224.json")
        if selected_artifacts[0] == "feature" and os.path.isfile(detector_path):
            artifact_stats.append((detector_path, os.stat(detector_path)))
        artifact_signature = tuple(
            (path, stat.st_mtime_ns, stat.st_size)
            for path, stat in artifact_stats
        )
        if (
            self._model_artifact_signature == artifact_signature
            and self.use_real_model
            and self.species_model is not None
            and self.health_model is not None
        ):
            return True

        with self._model_load_lock:
            if (
                self._model_artifact_signature == artifact_signature
                and self.use_real_model
                and self.species_model is not None
                and self.health_model is not None
            ):
                return True

            try:
                if selected_artifacts[0] == "feature":
                    feature_extractor = keras.models.load_model(selected_artifacts[1])
                    species_model = joblib.load(selected_artifacts[2])
                    health_model = joblib.load(selected_artifacts[3])
                    unknown_detector = None
                    if os.path.isfile(detector_path):
                        with open(detector_path, "r", encoding="utf-8") as detector_file:
                            unknown_detector = ValidationCalibratedDetector.from_dict(
                                json.load(detector_file)
                            )
                    input_size = feature_extractor.input_shape[1:3]
                    if all(size is not None for size in input_size):
                        self.img_size = tuple(int(size) for size in input_size)
                    self.species_feature_extractor = feature_extractor
                    self.unknown_detector = unknown_detector
                else:
                    species_model = keras.models.load_model(selected_artifacts[1])
                    health_model = keras.models.load_model(selected_artifacts[2])
                    input_size = species_model.input_shape[1:3]
                    if all(size is not None for size in input_size):
                        self.img_size = tuple(int(size) for size in input_size)
                    self.species_feature_extractor = None
                    self.unknown_detector = None

                self.species_model = species_model
                self.health_model = health_model
                self._model_artifact_signature = artifact_signature
                self.use_real_model = True
                logger.info("Loaded trained image classifiers from saved model artifacts")
                return True
            except Exception:
                logger.exception("Failed to load saved trained model artifacts")
                return bool(
                    self.use_real_model
                    and self.species_model is not None
                    and self.health_model is not None
                )

    def get_available_species(self):
        """Get list of available species from dataset directory"""
        species_list = []
        if os.path.exists(DATASET_DIR):
            for item in os.listdir(DATASET_DIR):
                item_path = os.path.join(DATASET_DIR, item)
                if os.path.isdir(item_path):
                    species_list.append(item)
        return sorted(species_list) if species_list else ['Cattleya', 'Dendrobium', 'Phalaenopsis', 'Vanda']
    
    def create_legacy_species_model(self):
        """Create transfer learning model using MobileNetV2 for better accuracy with small dataset"""
        try:
            # Use MobileNetV2 as base model with pre-trained weights
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(128, 128, 3),
                include_top=False,
                weights='imagenet',
                pooling='avg'
            )
            
            # Freeze the base model initially
            base_model.trainable = False
            self.species_base_model = base_model
            
            model = models.Sequential([
                layers.Input(shape=(128, 128, 3)),
                layers.Rescaling(scale=2.0, offset=-1.0),
                base_model,
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(len(self.species_classes), activation='softmax')  # Dynamic number of species
            ])
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            
            logger.info("Created transfer learning model with MobileNetV2")
            return model
            
        except Exception as e:
            logger.warning(f"Failed to create transfer learning model, falling back to CNN: {e}")
            # Fallback to simplified CNN
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.3),

                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(len(self.species_classes), activation='softmax')
            ])

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            return model
    
    def create_legacy_health_model(self):
        """Build a small-data health classifier on frozen ImageNet features."""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(128, 128, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        base_model.trainable = False
        self.health_base_model = base_model

        model = models.Sequential([
            layers.Input(shape=(128, 128, 3)),
            layers.Rescaling(scale=2.0, offset=-1.0),
            base_model,
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(2, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info("Created transfer-learning model for health classification")
        return model
    
    def create_species_model(self):
        """Create a frozen ImageNet feature extractor and small-data classifier."""
        existing_extractor_path = os.path.join(
            BASE_DIR,
            "models",
            "species_feature_extractor.keras"
        )
        if os.path.exists(existing_extractor_path):
            existing_extractor = keras.models.load_model(existing_extractor_path)
            self.species_feature_extractor = tf.keras.applications.MobileNetV2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights=None,
                pooling='avg'
            )
            self.species_feature_extractor.set_weights(existing_extractor.get_weights())
        else:
            self.species_feature_extractor = tf.keras.applications.MobileNetV2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet',
                pooling='avg'
            )
        self.species_feature_extractor.trainable = False
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
        )

    def create_health_model(self):
        """Create a class-balanced classifier for frozen image features."""
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
        )

    def create_training_augmentation_layers(self):
        """Return the exact ordered augmentation layers used for classifier training."""
        return [
            (
                "Rotation",
                layers.RandomRotation(0.1),
                "RandomRotation(0.1): randomly turns the image up to ±36° around its center. "
                "The layer fills any newly exposed corners using its default fill mode."
            ),
            (
                "Horizontal and vertical flips",
                layers.RandomFlip("horizontal_and_vertical"),
                "RandomFlip(horizontal_and_vertical): randomly mirrors the image horizontally "
                "and/or vertically, preserving the leaf label."
            ),
            (
                "Zoom",
                layers.RandomZoom(
                    height_factor=0.1,
                    width_factor=0.1,
                ),
                "RandomZoom(0.1): randomly zooms in or out by up to 10% in image height "
                "and width. Exposed edges use the layer's default fill mode."
            ),
            (
                "Brightness",
                layers.RandomBrightness(factor=0.15, value_range=(0, 1)),
                "RandomBrightness(0.15): randomly brightens or darkens normalized pixel "
                "values by up to 15%; outputs are clipped to the [0, 1] range."
            )
        ]

    def augment_training_images(self, images, copies=2):
        """Create randomized training-only variants of normalized RGB images."""
        augmentation = keras.Sequential([
            layer for _, layer, _ in self.create_training_augmentation_layers()
        ])
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        augmented = [
            tf.clip_by_value(augmentation(images, training=True), 0.0, 1.0).numpy()
            for _ in range(copies)
        ]
        return np.concatenate(augmented, axis=0)

    def create_training_augmentation_previews(self, image):
        """Render cumulative examples using the same layers as model training."""
        resized = image.convert("RGB").resize(self.img_size)
        images = tf.convert_to_tensor(
            np.asarray(resized, dtype=np.float32)[None, ...] / 255.0,
            dtype=tf.float32
        )
        previews = []
        for name, layer, configuration in self.create_training_augmentation_layers():
            images = layer(images, training=True)
            display_image = tf.cast(
                tf.round(tf.clip_by_value(images[0], 0.0, 1.0) * 255.0),
                tf.uint8
            ).numpy()
            buffer = BytesIO()
            Image.fromarray(display_image).save(buffer, format="PNG")
            preview_data = base64.b64encode(buffer.getvalue()).decode("ascii")
            previews.append({
                "name": name,
                "configuration": configuration,
                "preview_image": f"data:image/png;base64,{preview_data}"
            })
        return previews

    def get_training_augmentation_previews(self, image):
        """Keep preview failures visible without turning a valid prediction into a failure."""
        try:
            return self.create_training_augmentation_previews(image), None
        except Exception as error:
            logger.exception("Training augmentation preview generation failed")
            return [], str(error)

    def extract_image_features(self, images, progress_range=None):
        """Extract MobileNetV2 features from RGB images normalized to [0, 1]."""
        mobilenet_input = tf.keras.applications.mobilenet_v2.preprocess_input(
            np.asarray(images, dtype=np.float32) * 255.0
        )
        callbacks = []
        if progress_range is not None:
            start_percentage, end_percentage = progress_range

            class FeatureExtractionProgressCallback(tf.keras.callbacks.Callback):
                def on_predict_batch_end(self, batch, logs=None):
                    global training_progress
                    steps = self.params.get("steps") or 1
                    fraction = min(batch + 1, steps) / steps
                    training_progress["percentage"] = start_percentage + (
                        end_percentage - start_percentage
                    ) * fraction
                    training_progress["message"] = (
                        f"Extracting CNN features - batch {batch + 1}/{steps}"
                    )

            callbacks.append(FeatureExtractionProgressCallback())
        # Use a larger batch size and the lighter default input size so feature
        # extraction stays responsive on modest hardware without changing the model logic.
        return self.species_feature_extractor.predict(
            mobilenet_input,
            batch_size=64,
            verbose=0,
            callbacks=callbacks
        )

    def remove_background(self, img):
        """Isolate the foreground plant against a black background."""
        try:
            return extract_foreground(img)
        except Exception as e:
            logger.warning(f"Background removal failed: {e}, using original image")
            return img
    
    def extract_texture_features(self, img):
        """Extract texture features using GLCM"""
        try:
            img_array = np.array(img)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate GLCM features
            glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                               symmetric=True, normed=True)
            
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            
            return {
                'contrast': float(contrast),
                'dissimilarity': float(dissimilarity),
                'homogeneity': float(homogeneity),
                'energy': float(energy),
                'correlation': float(correlation)
            }
        except Exception as e:
            logger.warning(f"Texture extraction failed: {e}")
            return None
    
    def extract_shape_features(self, img):
        """Extract shape features from image"""
        try:
            img_array = np.array(img)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                contour = max(contours, key=cv2.contourArea)
                
                # Calculate features
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                else:
                    circularity = 0
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                extent = area / (w * h) if w * h > 0 else 0
                
                return {
                    'area': float(area),
                    'perimeter': float(perimeter),
                    'circularity': float(circularity),
                    'aspect_ratio': float(aspect_ratio),
                    'extent': float(extent)
                }
        except Exception as e:
            logger.warning(f"Shape extraction failed: {e}")
            return None
    
    def analyze_visible_symptoms(self, img):
        """Report visible color patterns without assigning a disease cause."""
        rgb = np.asarray(img.convert("RGB").resize((224, 224)))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(hsv)

        foreground = np.max(rgb, axis=2) > 20
        foreground_pixels = int(np.count_nonzero(foreground))
        if foreground_pixels < 100:
            return None

        def visible_area_percent(mask):
            component_count, _, stats, _ = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8),
                connectivity=8,
            )
            if component_count <= 1:
                return 0.0
            minimum_component_size = max(25, int(foreground_pixels * 0.005))
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            visible_pixels = int(
                np.sum(component_areas[component_areas >= minimum_component_size])
            )
            return 100.0 * visible_pixels / foreground_pixels

        yellow_areas = (
            foreground
            & (hue >= 18)
            & (hue <= 38)
            & (saturation >= 35)
            & (value >= 70)
        )
        brown_or_dark_areas = (
            foreground
            & (((hue <= 25) | (hue >= 170)) & (saturation >= 45) & (value <= 180))
        )

        return {
            "yellow_percent": float(visible_area_percent(yellow_areas)),
            "brown_dark_percent": float(visible_area_percent(brown_or_dark_areas))
        }

    def get_detailed_recommendations(
        self,
        species,
        health_status,
        species_confidence,
        health_confidence,
        symptom_observations=None,
        using_real_model=True,
        species_tentative=False
    ):
        """Build screening guidance from this image's model scores and measured color cues."""
        if species == "Unknown / uncertain":
            return (
                "Unable to confidently identify this image as a supported orchid category. "
                "Upload a sharp, well-lit photo showing one whole leaf."
            )
        if health_status == "unknown":
            return (
                "The orchid category was identified, but its health status is uncertain. "
                "Upload a sharp, well-lit photo. Do not treat this result as "
                "a confirmed disease diagnosis."
            )

        if symptom_observations:
            yellow_percent = symptom_observations["yellow_percent"]
            brown_dark_percent = symptom_observations["brown_dark_percent"]
        else:
            yellow_percent = 0.0
            brown_dark_percent = 0.0

        lines = []
        if not using_real_model:
            lines.append(
                "Model status: no trained model is available. The species and health "
                "labels shown above are rough color-based estimates, not reliable "
                "predictions."
            )

        is_healthy = health_status.replace("_", " ").casefold() == "healthy"
        has_yellowing = not is_healthy and yellow_percent >= 3.0
        has_brown_dark_areas = not is_healthy and brown_dark_percent >= 2.0

        care_actions = []
        if is_healthy:
            care_actions.extend(
                [
                    "Healthy result: Continue routine orchid care. Keep the plant in bright, "
                    "indirect light and water according to the potting medium and the needs "
                    "of this orchid; do not let the pot stand in water.",
                    "Check the upper and lower leaf surfaces regularly. If a new mark appears "
                    "or changes, photograph it in even light and monitor whether it spreads.",
                ]
            )
        else:
            care_actions.extend(
                [
                    "Not-healthy result: Inspect both sides of the leaves and the roots for "
                    "spots, pests, soft areas, or changes in texture.",
                    "Keep water off affected foliage, provide bright, indirect light, and "
                    "avoid sudden watering or fertilizer changes while you check the plant.",
                    "If an area is soft, wet, spreading, or enlarging, or roots are mushy, "
                    "seek an orchid-care assessment promptly. If no change is visible, retake "
                    "a clear photo in even light and monitor it.",
                ]
            )

        if has_yellowing:
            care_actions.append(
                "Yellowing detected: Note whether it is limited to one older leaf or also "
                "affects other leaves or new growth. Slow yellowing on one oldest leaf, "
                "with the rest of the plant firm, can be monitored. Spreading yellowing "
                "or yellowing on new growth calls for checking that the potting medium "
                "drains and inspecting roots for soft or mushy areas; seek an orchid-care "
                "assessment if roots feel soft. Keep the plant in bright, indirect light "
                "and avoid sudden watering or fertilizer changes."
            )
        if has_brown_dark_areas:
            care_actions.append(
                "Dark or brown areas detected: Keep the leaf dry and monitor any dry, "
                "unchanged marks. Soft, wet, or enlarging areas need prompt orchid-care "
                "assessment. Keep water off the foliage."
            )
        if not using_real_model:
            lines.append(
                "Image quality follow-up: use a sharp photo of the whole leaf in even, "
                "neutral light when comparing visible changes. Do not rely on the fallback "
                "labels to identify the orchid or assess its health."
            )
        if not using_real_model:
            lines.append(
                "Limit: image color measurements are visual screening cues only. They cannot "
                "identify the cause of a symptom or confirm that a plant is healthy or diseased."
            )
        lines.append(
            "What to do\n"
            + "\n".join(
                f"{index}. {action}"
                for index, action in enumerate(care_actions, start=1)
            )
        )
        return "\n".join(lines)
    
    def analyze_image(self, image_data):
        try:
            if self.load_saved_models():
                logger.info("Using real ML model for prediction")
                return self.predict_with_model(image_data)
            logger.error("Dataset-trained image classifiers are not available for prediction")
            return {
                "success": False,
                "error": (
                    "Dataset-based analysis is temporarily unavailable. "
                    "Please ask the administrator to check the trained model."
                ),
            }
        except Exception as e:
            logger.exception("Image analysis failed")
            return {"success": False, "error": str(e)}
    
    def fallback_analysis(self, image_data):
        """Fallback color-based analysis when model is not trained"""
        try:
            if isinstance(image_data, str) and image_data.startswith("data:image"):
                _, encoded = image_data.split(",", 1)
                img = ImageOps.exif_transpose(
                    Image.open(BytesIO(base64.b64decode(encoded)))
                )
            else:
                img = ImageOps.exif_transpose(Image.open(image_data))

            # Apply background removal for display
            img_processed = self.remove_background(img)
            
            # Convert preprocessed image to base64 for display
            buffered = BytesIO()
            img_processed.save(buffered, format="JPEG")
            preprocessed_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            preprocessed_image_url = f"data:image/jpeg;base64,{preprocessed_image_base64}"

            img_resized = img.convert("RGB").resize(self.img_size)
            normalized_input = np.asarray(img_resized, dtype=np.float32) / 255.0
            arr = np.asarray(img_resized)
            model_input_buffer = BytesIO()
            img_resized.save(model_input_buffer, format="PNG")
            model_input_preview = (
                "data:image/png;base64,"
                + base64.b64encode(model_input_buffer.getvalue()).decode("ascii")
            )
            normalized_range = [
                float(np.min(normalized_input)),
                float(np.max(normalized_input))
            ]
            rgb_channel_averages = {
                "R": float(np.mean(arr[:, :, 0])),
                "G": float(np.mean(arr[:, :, 1])),
                "B": float(np.mean(arr[:, :, 2]))
            }
            
            r = rgb_channel_averages["R"]
            g = rgb_channel_averages["G"]
            b = rgb_channel_averages["B"]
            
            ci = ((g - r) / (g + r) + 1) / 2 * 100 if (g + r) > 0 else 50
            ci = max(0, min(100, ci))
            
            is_healthy = ci > 50
            health_status = "healthy" if is_healthy else "not_healthy"
            
            if ci > 70:
                species = "Phalaenopsis"
                confidence = 85.0
            elif ci > 60:
                species = "Cattleya"
                confidence = 80.0
            elif ci > 50:
                species = "Dendrobium"
                confidence = 75.0
            else:
                species = "Vanda"
                confidence = 70.0
            
            symptom_observations = self.analyze_visible_symptoms(img_processed)
            recommendation = self.get_detailed_recommendations(
                species,
                health_status,
                confidence,
                confidence,
                symptom_observations,
                using_real_model=False
            )
            augmentation_previews, augmentation_preview_error = (
                self.get_training_augmentation_previews(img_resized)
            )
            
            return {
                "success": True,
                "species": species,
                "confidence": None,
                "health_status": health_status,
                "is_healthy": is_healthy,
                "color_index": round(ci, 1),
                "recommendation": recommendation,
                "green_value": round(g, 1),
                "red_value": round(r, 1),
                "blue_value": round(b, 1),
                "using_real_model": False,
                "preprocessed_image": preprocessed_image_url,
                "model_input_image": model_input_preview,
                "normalized_range": normalized_range,
                "model_input_range": normalized_range,
                "model_input_transform": "RGB float32 values divided by 255; range 0–1.",
                "rgb_channel_averages": rgb_channel_averages,
                "training_augmentation_previews": augmentation_previews,
                "training_augmentation_preview_error": augmentation_preview_error,
                "classifier_pipeline": [
                    "224 × 224 RGB input, normalized to 0–1",
                    "Color-based fallback estimate",
                    "Species and health screening labels"
                ],
                "visible_symptoms": symptom_observations
            }
        except Exception as e:
            logger.exception("Fallback analysis failed")
            return {"success": False, "error": str(e)}
    
    def predict_with_model(self, image_data):
        """Use trained CNN model for prediction with enhanced preprocessing"""
        try:
            if isinstance(image_data, str) and image_data.startswith("data:image"):
                _, encoded = image_data.split(",", 1)
                img = ImageOps.exif_transpose(
                    Image.open(BytesIO(base64.b64decode(encoded)))
                )
            else:
                img = ImageOps.exif_transpose(Image.open(image_data))

            if min(img.size) < 64:
                return {
                    "success": True,
                    "species": "Unknown / uncertain",
                    "confidence": None,
                    "species_confidence": None,
                    "health_confidence": None,
                    "uncertainty_calibrated": self.unknown_detector is not None,
                    "health_status": "unknown",
                    "is_healthy": None,
                    "unknown_or_uncertain": True,
                    "out_of_distribution": False,
                    "warning": (
                        "The image is too small for reliable identification. Upload a "
                        "sharp, well-lit photo showing one whole leaf."
                    ),
                    "recommendation": (
                        "Unable to analyze this image reliably. Upload a larger, sharp "
                        "photo showing one whole leaf."
                    ),
                    "using_real_model": True,
                }

            # Apply background removal
            img_processed = self.remove_background(img)
            
            # Convert preprocessed image to base64 for display
            buffered = BytesIO()
            img_processed.save(buffered, format="JPEG")
            preprocessed_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            preprocessed_image_url = f"data:image/jpeg;base64,{preprocessed_image_base64}"
            
            # Preserve the full frame and add a centered square view for species matching.
            logger.info(f"Image before resize: {img.size}")
            rgb_image = img.convert("RGB")
            img_resized = rgb_image.resize(self.img_size)
            width, height = rgb_image.size
            crop_side = min(width, height)
            square_species_view = rgb_image.crop((
                (width - crop_side) // 2,
                (height - crop_side) // 2,
                (width + crop_side) // 2,
                (height + crop_side) // 2,
            )).resize(self.img_size)
            species_views = [img_resized]
            if width != height:
                species_views.append(square_species_view)
            sharpness = (
                self.unknown_detector.image_sharpness(np.asarray(img_resized))
                if self.unknown_detector is not None
                else None
            )
            logger.info(f"Image after resize: {img_resized.size}")
            species_images = np.clip(
                np.asarray([
                    np.asarray(view, dtype=np.float32) / 255.0
                    for view in species_views
                ]),
                0,
                1,
            )
            img_array = species_images[:1]
            model_input_buffer = BytesIO()
            img_resized.save(model_input_buffer, format="PNG")
            model_input_preview = (
                "data:image/png;base64,"
                + base64.b64encode(model_input_buffer.getvalue()).decode("ascii")
            )
            species_crop_buffer = BytesIO()
            square_species_view.save(species_crop_buffer, format="PNG")
            species_crop_preview = (
                "data:image/png;base64,"
                + base64.b64encode(species_crop_buffer.getvalue()).decode("ascii")
            )
            normalized_range = [
                float(np.min(img_array)),
                float(np.max(img_array))
            ]
            model_tensor = np.asarray([
                tf.keras.applications.mobilenet_v2.preprocess_input(view * 255.0)
                for view in species_images
            ]) if self.species_feature_extractor is not None else species_images
            model_input_range = [
                float(np.min(model_tensor)),
                float(np.max(model_tensor))
            ]
            model_input_transform = (
                "Species classification averages the full-frame and centered square "
                "views. Both use RGB float32 values divided by 255, then MobileNetV2 "
                "preprocess_input maps values to [-1, 1]. Health classification uses "
                "the full-frame view."
                if self.species_feature_extractor is not None
                else "Species classification averages the full-frame and centered square "
                "views; both use RGB float32 values divided by 255. Health classification "
                "uses the full-frame view."
            )
            rgb_channel_averages = {
                "R": float(np.mean(img_resized, axis=(0, 1))[0]),
                "G": float(np.mean(img_resized, axis=(0, 1))[1]),
                "B": float(np.mean(img_resized, axis=(0, 1))[2])
            }
            
            logger.info(f"Image array shape: {img_array.shape}")
            logger.info(f"Final input shape: {img_array.shape}")
            
            feature_vector = None
            if self.species_feature_extractor is not None:
                species_features = self.extract_image_features(species_images)
                feature_vector = species_features[:1]

            if feature_vector is not None and hasattr(self.species_model, "predict_proba"):
                species_view_predictions = self.species_model.predict_proba(species_features)
                calibration_species_pred = species_view_predictions[0]
                species_pred = np.mean(species_view_predictions, axis=0, keepdims=True)
                health_pred = self.health_model.predict_proba(feature_vector)
            else:
                species_view_predictions = self.species_model.predict(
                    species_images,
                    verbose=0
                )
                calibration_species_pred = species_view_predictions[0]
                species_pred = np.mean(species_view_predictions, axis=0, keepdims=True)
                health_pred = (
                    self.health_model.predict_proba(feature_vector)
                    if feature_vector is not None and hasattr(self.health_model, "predict_proba")
                    else self.health_model.predict(img_array, verbose=0)
                )
            score_calibration = {
                "species": False,
                "health": False,
            }
            if self.unknown_detector is not None:
                calibration_species_pred = self.unknown_detector.calibrate_probabilities(
                    np.asarray(calibration_species_pred).reshape(1, -1),
                    "species",
                )[0]
                species_pred = self.unknown_detector.calibrate_probabilities(
                    species_pred,
                    "species",
                )
                health_pred = self.unknown_detector.calibrate_probabilities(
                    health_pred,
                    "health",
                )
                score_calibration = {
                    target: self.unknown_detector.has_probability_calibration(target)
                    for target in ("species", "health")
                }
            species_idx = np.argmax(species_pred[0])
            species_confidence = float(species_pred[0][species_idx] * 100)
            predicted_species = self.species_classes[species_idx]
            
            logger.info(f"Species predictions: {species_pred[0]}")
            logger.info(f"Species classes: {self.species_classes}")
            logger.info(
                "Predicted species: %s (confidence: %.2f%%)",
                predicted_species,
                species_confidence,
            )
            
            health_idx = np.argmax(health_pred[0])
            health_confidence = float(health_pred[0][health_idx] * 100)
            predicted_health = self.health_classes[health_idx]
            
            logger.info(f"Health predictions: {health_pred[0]}")
            logger.info(f"Health classes: {self.health_classes}")
            logger.info(
                "Predicted health: %s (confidence: %.2f%%)",
                predicted_health,
                health_confidence,
            )

            warning = None
            out_of_distribution = False
            too_blurry = False
            species_tentative = False
            feature_distance = None
            if feature_vector is None or self.unknown_detector is None:
                species = predicted_species
                health_status = predicted_health
                warning = (
                    f"Provisional model prediction: {species}, "
                    f"{health_status.replace('_', ' ')}. Scores are not calibrated, "
                    "and unsupported-image detection is unavailable. An unfamiliar "
                    "orchid or non-orchid may be assigned a supported label. Retrain "
                    "the model before relying on these results."
                )
            else:
                candidate_species_set = self.unknown_detector.prediction_set(
                    calibration_species_pred,
                    "species",
                )
                out_of_distribution, feature_distance = (
                    self.unknown_detector.is_out_of_distribution(
                        feature_vector[0],
                        predicted_species,
                    )
                )
                too_blurry = self.unknown_detector.is_too_blurry(sharpness)
                if too_blurry:
                    species = "Unknown / uncertain"
                    warning = (
                        "Unable to confidently identify this orchid leaf. The photo appears "
                        "blurrier than the training validation images. Upload a sharp, "
                        "well-lit photo showing one whole leaf."
                    )
                else:
                    species = predicted_species
                    species_tentative = (
                        out_of_distribution
                        or predicted_species not in candidate_species_set
                        or len(candidate_species_set) > 1
                    )

                candidate_health_set = self.unknown_detector.prediction_set(
                    health_pred[0],
                    "health",
                )
                health_status = predicted_health
                if out_of_distribution or too_blurry or len(candidate_health_set) != 1:
                    if warning is None:
                        warning = (
                            f"The model classified this image as "
                            f"{health_status.replace('_', ' ')}, but validation checks "
                            "could not confirm the health label. Retake a sharp, well-lit photo."
                        )

            is_healthy = health_status == "healthy"
            overall_confidence = (species_confidence + health_confidence) / 2
            
            # Extract additional features for detailed analysis
            texture_features = self.extract_texture_features(img_processed)
            shape_features = self.extract_shape_features(img_processed)
            
            # Generate detailed recommendations
            symptom_observations = self.analyze_visible_symptoms(img_processed)
            recommendation = self.get_detailed_recommendations(
                species,
                health_status,
                species_confidence,
                health_confidence,
                symptom_observations,
                species_tentative=species_tentative
            )
            augmentation_previews, augmentation_preview_error = (
                self.get_training_augmentation_previews(img_resized)
            )
            
            return {
                "success": True,
                "species": species,
                "confidence": overall_confidence,
                "health_status": health_status,
                "health_prediction": predicted_health,
                "is_healthy": is_healthy,
                "recommendation": recommendation,
                "species_confidence": species_confidence,
                "health_confidence": health_confidence,
                "species_score_calibrated": score_calibration["species"],
                "health_score_calibrated": score_calibration["health"],
                "uncertainty_calibrated": self.unknown_detector is not None,
                "unknown_or_uncertain": bool(warning) or species_tentative,
                "warning": warning,
                "out_of_distribution": out_of_distribution,
                "too_blurry": too_blurry,
                "image_sharpness": sharpness,
                "feature_distance": feature_distance,
                "using_real_model": True,
                "preprocessed_image": preprocessed_image_url,
                "model_input_image": model_input_preview,
                "species_crop_image": species_crop_preview,
                "species_view_count": len(species_views),
                "normalized_range": normalized_range,
                "model_input_range": model_input_range,
                "model_input_transform": model_input_transform,
                "rgb_channel_averages": rgb_channel_averages,
                "training_augmentation_previews": augmentation_previews,
                "training_augmentation_preview_error": augmentation_preview_error,
                "classifier_pipeline": (
                    [
                        "Average full-frame and centered square species views",
                        "Full-frame 224 × 224 input used for health classification",
                        "Frozen MobileNetV2 feature extractor",
                        "Species and health classifiers"
                    ]
                    if feature_vector is not None
                    else [
                        "Average full-frame and centered square species views",
                        "Full-frame 224 × 224 input used for health classification",
                        "Trained image model",
                        "Species and health predictions"
                    ]
                ),
                "texture_features": texture_features,
                "shape_features": shape_features,
                "visible_symptoms": symptom_observations
            }
        except Exception as e:
            logger.exception("Trained model prediction failed")
            return {
                "success": False,
                "error": "The trained image analysis failed. Please try again.",
            }

orchid_ai = OrchidAI()

# =========================================================
# ROUTES
# =========================================================
@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        logger.info(f"Login attempt for username: {username}")
        
        # Check admin credentials
        if ADMIN_PASSWORD and username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["logged_in"] = True
            session["user_type"] = "admin"
            logger.info("Admin login successful")
            return redirect(url_for("admin_dashboard"))
        
        # Check user credentials from users.json
        users_file = os.path.join(BASE_DIR, "users.json")
        users = load_app_document("users", users_file, {})

        logger.info(f"Available users: {list(users.keys())}")

        if username in users:
            logger.info("Username found in stored users")
            if users[username]["password"] == password:
                session["logged_in"] = True
                session["user_type"] = "user"
                session["username"] = username
                logger.info("User login successful")
                return redirect(url_for("user_dashboard"))
            else:
                logger.info("Password mismatch")
        else:
            logger.info("Username not found in stored users")
        
        return render_template("login.html", app_name=APP_NAME, error="Invalid username or password")
    
    return render_template("login.html", app_name=APP_NAME)

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))

@app.route("/create-account", methods=["POST"])
def create_account():
    username = request.form.get("username")
    password = request.form.get("password")
    confirm_password = request.form.get("confirm_password")
    
    # Validate input
    if not username or not password or not confirm_password:
        return render_template("login.html", app_name=APP_NAME, error="All fields are required")
    
    if password != confirm_password:
        return render_template("login.html", app_name=APP_NAME, error="Passwords do not match")
    
    if len(password) < 6:
        return render_template("login.html", app_name=APP_NAME, error="Password must be at least 6 characters")
    
    # Check if username already exists (for demo, just check against admin)
    if username == ADMIN_USERNAME:
        return render_template("login.html", app_name=APP_NAME, error="Username already exists")
    
    users_file = os.path.join(BASE_DIR, "users.json")
    users = load_app_document("users", users_file, {})
    
    if username in users:
        return render_template("login.html", app_name=APP_NAME, error="Username already exists")
    
    users[username] = {
        "password": password,
        "created_at": datetime.now().isoformat()
    }
    save_app_document("users", users)
    
    logger.info(f"New account created: {username}")
    
    return render_template("login.html", app_name=APP_NAME, success="Account created successfully! Please login.")

@app.route("/home")
def home():
    return render_template("mobile_index.html", app_name=APP_NAME)

@app.route("/camera")
def camera():
    return render_template("mobile_camera.html", app_name=APP_NAME)

@app.route("/upload")
def upload():
    return render_template("mobile_upload.html", app_name=APP_NAME)

@app.route("/about")
def about():
    return render_template("mobile_about.html", app_name=APP_NAME)

@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    return render_template("admin.html", app_name=APP_NAME)

@app.route("/user/dashboard")
@login_required
def user_dashboard():
    return render_template("user_dashboard.html", app_name=APP_NAME)

@app.route("/dataset")
@login_required
def dataset():
    return render_template("dataset_overview.html", app_name=APP_NAME)

@app.route("/predictions")
@login_required
def predictions():
    return render_template("predictions.html", app_name=APP_NAME)

@app.route("/data_training")
@login_required
def data_training():
    return render_template("data_training.html", app_name=APP_NAME)

@app.route("/data_augmentation")
@login_required
def data_augmentation():
    return render_template("data_augmentation.html", app_name=APP_NAME)

@app.route("/preprocessing")
@login_required
def preprocessing():
    return render_template("preprocessing.html", app_name=APP_NAME)

@app.route("/image_pipeline")
@login_required
def image_pipeline():
    return render_template("image_pipeline.html", app_name=APP_NAME)

@app.route("/performance")
@login_required
def performance():
    return render_template("performance.html", app_name=APP_NAME)

@app.route("/advanced_eda")
@login_required
def advanced_eda():
    return render_template("advanced_eda.html", app_name=APP_NAME)

@app.route("/image-processing")
@login_required
def image_processing():
    """Image Processing Visualization Page"""
    return render_template("image_processing.html", app_name=APP_NAME)

# =========================================================
# API ROUTES
# =========================================================
@app.route("/api/dataset/samples", methods=["GET"])
def api_dataset_samples():
    samples = get_dataset_samples()
    stats = get_dataset_stats()
    return jsonify({
        "success": True,
        "samples": samples,
        "stats": stats
    })

@app.route("/api/admin/stats", methods=["GET"])
@login_required
def api_admin_stats():
    """Get dashboard statistics"""
    try:
        total_images = 0
        healthy_images = 0
        unhealthy_images = 0
        species_counts = {}

        if os.path.exists(DATASET_DIR):
            species_list = sorted(
                name for name in os.listdir(DATASET_DIR)
                if os.path.isdir(os.path.join(DATASET_DIR, name))
            )
        else:
            species_list = []

        for species in species_list:
            species_path = os.path.join(DATASET_DIR, species)
            species_total = 0

            healthy_path = os.path.join(species_path, 'healthy')
            if os.path.exists(healthy_path):
                healthy_images_list = glob.glob(os.path.join(healthy_path, '*.*'))
                healthy_images_list = [f for f in healthy_images_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
                healthy_count = len(healthy_images_list)
                healthy_images += healthy_count
                total_images += healthy_count
                species_total += healthy_count
                logger.info(f"{species} healthy: {healthy_count}")

            for health_folder in ['not healthy', 'not_healthy']:
                not_healthy_path = os.path.join(species_path, health_folder)
                if os.path.exists(not_healthy_path):
                    unhealthy_images_list = glob.glob(os.path.join(not_healthy_path, '*.*'))
                    unhealthy_images_list = [f for f in unhealthy_images_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
                    unhealthy_count = len(unhealthy_images_list)
                    unhealthy_images += unhealthy_count
                    total_images += unhealthy_count
                    species_total += unhealthy_count
                    logger.info(f"{species} {health_folder}: {unhealthy_count}")

            if species_total > 0 or os.path.isdir(species_path):
                species_counts[species] = species_total

        return jsonify({
            "success": True,
            "total_images": total_images,
            "species_count": len(species_counts),
            "healthy_images": healthy_images,
            "unhealthy_images": unhealthy_images,
            "species_counts": species_counts,
            "model_accuracy": "--",
            "last_training_date": "--"
        })
    except Exception as e:
        logger.exception("Failed to get admin stats")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/admin/users", methods=["GET"])
@login_required
def api_admin_users():
    """Get user list"""
    return jsonify({
        "success": True,
        "users": [
            {"username": "admin", "role": "Administrator", "status": "Active"}
        ]
    })

@app.route("/api/admin/scan-records", methods=["GET"])
@login_required
def api_scan_records():
    """Get scan records"""
    try:
        records = load_scan_records()
        stats = get_scan_statistics()

        # Get filter parameters
        species_filter = request.args.get('species')
        health_filter = request.args.get('health')
        limit = int(request.args.get('limit', 50))

        # Apply filters
        filtered_records = records
        if species_filter:
            filtered_records = [r for r in filtered_records if r.get('species') == species_filter]
        if health_filter:
            filtered_records = [r for r in filtered_records if r.get('health') == health_filter]

        # Limit results
        recent_scans = filtered_records[:limit]

        return jsonify({
            "success": True,
            "total_scans": stats["total_scans"],
            "today_scans": stats["today_scans"],
            "week_scans": stats["week_scans"],
            "recent_scans": recent_scans
        })
    except Exception as e:
        logger.exception("Failed to get scan records")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/admin/upload-dataset", methods=["POST"])
@login_required
def api_upload_dataset():
    """Upload dataset images"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})
        
        species = request.form.get('species')
        health = request.form.get('health', 'healthy')

        if not species:
            return jsonify({"success": False, "error": "Species not specified"})

        # Validate species name (allow dynamic species)
        if not species.replace(' ', '').isalnum():
            return jsonify({"success": False, "error": "Invalid species name. Use only letters and spaces."})

        # Capitalize species name properly
        species = ' '.join(word.capitalize() for word in species.split())
        
        # Normalize health status to match folder naming
        if health in ['not_healthy', 'not healthy']:
            health = 'not healthy'  # Use space version as default
        elif health != 'healthy':
            return jsonify({"success": False, "error": "Invalid health status. Use 'healthy' or 'not healthy'"})
        
        # Create directory if it doesn't exist
        species_dir = os.path.join(DATASET_DIR, species, health)
        os.makedirs(species_dir, exist_ok=True)
        
        # Save file
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({"success": False, "error": "Invalid filename"})
        file_path = os.path.join(species_dir, filename)
        if os.path.exists(file_path):
            stem, extension = os.path.splitext(filename)
            filename = f"{stem}_{uuid.uuid4().hex[:8]}{extension}"
            file_path = os.path.join(species_dir, filename)
        file.save(file_path)
        image_path = os.path.relpath(file_path, DATASET_DIR).replace(os.sep, "/")
        orchid_ai.species_classes = orchid_ai.get_available_species()
        register_uploaded_image_group(image_path)
        mark_dataset_image_as_new_upload(image_path)
        
        return jsonify({
            "success": True,
            "message": "File uploaded successfully",
            "filename": filename,
            "image_path": image_path,
        })
    except Exception as e:
        logger.exception("Dataset upload failed")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/admin/add-species", methods=["POST"])
@login_required
def api_add_species():
    """Add a new orchid species/family to the system"""
    try:
        data = request.get_json()
        species_name = (data.get('species_name') or '').strip()

        if not species_name:
            return jsonify({"success": False, "error": "Species name is required"})

        if not species_name.replace(' ', '').isalnum():
            return jsonify({"success": False, "error": "Species name must contain only letters and spaces"})

        formatted_name = ' '.join(word[:1].upper() + word[1:] for word in species_name.split())

        species_path = os.path.join(DATASET_DIR, formatted_name)
        if os.path.exists(species_path):
            return jsonify({"success": False, "error": f"Species '{formatted_name}' already exists"})

        # Create species directory structure
        os.makedirs(species_path, exist_ok=True)
        os.makedirs(os.path.join(species_path, 'healthy'), exist_ok=True)
        os.makedirs(os.path.join(species_path, 'not healthy'), exist_ok=True)

        logger.info(f"Created new species: {formatted_name}")

        return jsonify({
            "success": True,
            "message": f"Species '{formatted_name}' added successfully",
            "species": formatted_name
        })
    except Exception as e:
        logger.exception("Failed to add species")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/admin/delete-species", methods=["POST"])
@login_required
def api_delete_species():
    """Delete a species directory and associated manifest entries"""
    try:
        data = request.get_json() or {}
        species_name = (data.get('species_name') or '').strip()

        if not species_name:
            return jsonify({"success": False, "error": "Species name is required"})

        formatted_name = ' '.join(word[:1].upper() + word[1:] for word in species_name.split())
        species_path = os.path.join(DATASET_DIR, formatted_name)

        if not os.path.exists(species_path):
            return jsonify({"success": False, "error": f"Species '{formatted_name}' was not found"})

        shutil.rmtree(species_path)

        manifest_path = os.path.join(DATASET_DIR, 'image_groups.csv')
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r', newline='') as file:
                reader = csv.DictReader(file)
                fieldnames = reader.fieldnames or ['image_path', 'plant_id', 'session_id']
                rows = [row for row in reader if row.get('image_path', '').replace('\\', '/').split('/')[0] != formatted_name]

            with open(manifest_path, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        logger.info(f"Deleted species: {formatted_name}")
        return jsonify({
            "success": True,
            "message": f"Species '{formatted_name}' deleted successfully",
            "species": formatted_name
        })
    except Exception as e:
        logger.exception("Failed to delete species")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/admin/list-species", methods=["GET"])
@login_required
def api_list_species():
    """Get list of all available species"""
    try:
        species_list = []
        if os.path.exists(DATASET_DIR):
            for item in os.listdir(DATASET_DIR):
                item_path = os.path.join(DATASET_DIR, item)
                if os.path.isdir(item_path):
                    # Count images in this species
                    total_images = 0
                    for health in ['healthy', 'not healthy', 'not_healthy']:
                        health_path = os.path.join(item_path, health)
                        if os.path.exists(health_path):
                            images = glob.glob(os.path.join(health_path, '*.*'))
                            images = [f for f in images if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
                            total_images += len(images)
                    
                    species_list.append({
                        "name": item,
                        "total_images": total_images
                    })

        return jsonify({"success": True, "species": species_list})
    except Exception as e:
        logger.exception("Failed to list species")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/admin/dataset-images", methods=["GET"])
@login_required
def api_dataset_images():
    """Get list of dataset images"""
    try:
        species_filter = request.args.get('species', '').strip()
        health_filter = request.args.get('health', '').strip()
        valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.webp')
        images = []

        if not os.path.isdir(DATASET_DIR):
            return jsonify({"success": True, "images": []})

        for species_entry in os.scandir(DATASET_DIR):
            if not species_entry.is_dir():
                continue
            species = species_entry.name
            if species_filter and species != species_filter:
                continue

            if health_filter == 'healthy':
                health_folders = ('healthy',)
            elif health_filter in ('not healthy', 'not_healthy'):
                health_folders = ('not healthy', 'not_healthy')
            elif health_filter:
                health_folders = ()
            else:
                health_folders = ('healthy', 'not healthy', 'not_healthy')

            for health_folder in health_folders:
                species_dir = os.path.join(species_entry.path, health_folder)
                if not os.path.isdir(species_dir):
                    continue

                for filename in os.listdir(species_dir):
                    if not filename.lower().endswith(valid_extensions):
                        continue

                    relative_path = os.path.join(species, health_folder, filename)
                    images.append({
                        "filename": filename,
                        "url": url_for("serve_dataset", filename=relative_path.replace(os.sep, "/")),
                        "species": species,
                        "health": "Healthy" if health_folder == "healthy" else "Not Healthy",
                        "health_folder": health_folder
                    })
        
        return jsonify({"success": True, "images": images})
    except Exception as e:
        logger.exception("Failed to get dataset images")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/admin/delete-dataset-image", methods=["POST"])
@login_required
def api_delete_dataset_image():
    """Delete a dataset image"""
    try:
        data = request.get_json()
        species = data.get('species')
        health = data.get('health', 'healthy')
        filename = data.get('filename')
        
        if not species or not filename:
            return jsonify({"success": False, "error": "Missing parameters"})
        
        file_path = os.path.join(DATASET_DIR, species, health, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({"success": True, "message": "File deleted successfully"})
        else:
            return jsonify({"success": False, "error": "File not found"})
    except Exception as e:
        logger.exception("Failed to delete dataset image")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/admin/recommendations", methods=["GET", "POST"])
@login_required
def api_recommendations():
    """Get or update recommendations"""
    recommendations_file = os.path.join(BASE_DIR, "recommendations.json")
    default_recommendations = {
        "healthy": [
            "Continue current watering schedule",
            "Maintain indirect sunlight",
            "Fertilize every 2 weeks",
            "Monitor for pests regularly"
        ],
        "unhealthy": [
            "Remove infected leaves immediately",
            "Apply appropriate fungicide",
            "Reduce watering frequency",
            "Improve air circulation",
            "Check for pests and treat accordingly"
        ]
    }
    
    if request.method == "GET":
        try:
            data = load_app_document("recommendations", recommendations_file, default_recommendations)
            return jsonify({"success": True, "recommendations": data})
        except Exception as e:
            logger.exception("Failed to get recommendations")
            return jsonify({"success": False, "error": str(e)})
    
    elif request.method == "POST":
        try:
            data = request.get_json()
            recommendations = data.get('recommendations')
            
            if not recommendations:
                return jsonify({"success": False, "error": "No recommendations provided"})
            
            save_app_document("recommendations", recommendations)
            
            return jsonify({"success": True, "message": "Recommendations updated successfully"})
        except Exception as e:
            logger.exception("Failed to update recommendations")
            return jsonify({"success": False, "error": str(e)})

@app.route("/api/admin/model-metrics", methods=["GET"])
@login_required
def api_model_metrics():
    """Get model training metrics"""
    metrics_file = os.path.join(BASE_DIR, "models", "training_metrics.json")
    default_metrics = {
        "accuracy": "--",
        "precision": "--",
        "recall": "--",
        "f1_score": "--",
        "confusion_matrix": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        "training_history": {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []},
        "last_training_date": "--"
    }
    
    try:
        if MYSQL_CONFIG["password"]:
            data = load_app_document("training_metrics", metrics_file, default_metrics)
        else:
            logger.warning(
                "MYSQL_PASSWORD is not set; serving model metrics from the local metrics file"
            )
            data = default_metrics
            if os.path.exists(metrics_file):
                with open(metrics_file, "r", encoding="utf-8") as metrics_data:
                    data = json.load(metrics_data)
        if isinstance(data, dict):
            evaluation_split = data.get("evaluation_split") or {}
            data = dict(data)
            if (
                data.get("status") == "training"
                and training_progress.get("status") != "training"
            ):
                data["status"] = "interrupted"
            try:
                training_scope = data.get("training_scope", "all")
                current_image_paths = get_training_image_paths(training_scope)
                current_group_ids, group_error = load_image_group_assignments(
                    current_image_paths,
                    allow_extra_paths=True,
                )
                data["evaluation_is_current"] = bool(
                    current_image_paths
                    and not group_error
                    and data.get("status") == "completed"
                    and data.get("group_manifest_verified") is True
                    and evaluation_split.get("group_manifest_verified") is True
                    and data.get("total_images") == len(current_image_paths)
                    and data.get("dataset_fingerprint")
                    == get_dataset_fingerprint(current_image_paths, current_group_ids)
                )
            except (OSError, ValueError) as error:
                logger.warning("Could not verify saved model metrics: %s", error)
                data["evaluation_is_current"] = False
        return jsonify({"success": True, "metrics": data})
    except Exception as e:
        logger.exception("Failed to get model metrics")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/admin/settings", methods=["GET", "POST"])
@login_required
def api_settings():
    """Get or update system settings"""
    settings_file = os.path.join(BASE_DIR, "settings.json")
    default_settings = {
        "app_name": APP_NAME,
        "version": "1.0.0",
        "max_upload_size": 16,
        "allowed_extensions": ["png", "jpg", "jpeg", "gif"],
        "model_threshold": 0.5,
        "auto_reload": True
    }
    
    if request.method == "GET":
        try:
            data = load_app_document("settings", settings_file, default_settings)
            return jsonify({"success": True, "settings": data})
        except Exception as e:
            logger.exception("Failed to get settings")
            return jsonify({"success": False, "error": str(e)})
    
    elif request.method == "POST":
        try:
            data = request.get_json()
            settings = data.get('settings')
            
            if not settings:
                return jsonify({"success": False, "error": "No settings provided"})
            
            save_app_document("settings", settings)
            
            return jsonify({"success": True, "message": "Settings updated successfully"})
        except Exception as e:
            logger.exception("Failed to update settings")
            return jsonify({"success": False, "error": str(e)})

@app.route("/api/admin/train-model", methods=["POST"])
@login_required
def api_train_model():
    """Start model training with the requested old/new dataset scope."""
    lock_acquired = False
    try:
        orchid_ai.species_classes = orchid_ai.get_available_species()
        payload = request.get_json(silent=True) or {}
        training_scope = payload.get("scope", "all")
        if training_scope not in {"old", "new", "all"}:
            return jsonify({
                "success": False,
                "error": "Choose old images, new uploads, or all images for training.",
            })

        all_image_paths = get_dataset_image_paths()
        image_paths = get_training_image_paths(training_scope)
        total_images = len(image_paths)
        if total_images < 10:
            scope_label = {
                "old": "old dataset",
                "new": "new uploads",
                "all": "dataset",
            }[training_scope]
            return jsonify({
                "success": False,
                "error": (
                    f"The selected {scope_label} has only {total_images} images. "
                    "At least 10 images are required to train."
                ),
            })

        if not os.path.exists(get_image_group_manifest_path()):
            create_image_group_manifest_template()

        try:
            repair_missing_group_manifest_ids(all_image_paths)
        except ValueError as error:
            return jsonify({
                "success": False,
                "error": f"Could not assign group IDs automatically: {error}",
            })

        image_group_ids, group_error = load_image_group_assignments(
            image_paths,
            allow_extra_paths=True,
        )
        if group_error:
            return jsonify({
                "success": False,
                "error": (
                    f"{group_error} Fill plant_id and/or session_id in "
                    "datasets/image_groups.csv, then start training again."
                )
            })
        try:
            species_labels, health_labels = get_dataset_labels_for_paths(image_paths)
            make_grouped_train_validation_test_split(
                species_labels,
                [image_group_ids[path] for path in image_paths],
                "species classification"
            )
            make_grouped_train_validation_test_split(
                health_labels,
                [image_group_ids[path] for path in image_paths],
                "health classification"
            )
        except ValueError as error:
            return jsonify({"success": False, "error": str(error)})

        if not training_lock.acquire(blocking=False):
            return jsonify({
                "success": False,
                "error": "Model training is already running. Wait for it to finish before starting another run.",
            }), 409
        lock_acquired = True
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(BASE_DIR, "models")
        os.makedirs(models_dir, exist_ok=True)
        dataset_fingerprint = get_dataset_fingerprint(image_paths, image_group_ids)
        
        # Update training metrics file to show training started
        metrics_file = os.path.join(models_dir, "training_metrics.json")
        training_metrics = {
            "status": "training",
            "total_images": total_images,
            "training_scope": training_scope,
            "dataset_fingerprint": dataset_fingerprint,
            "group_manifest_verified": False,
            "start_time": datetime.now().isoformat(),
            "accuracy": "--",
            "precision": "--",
            "recall": "--",
            "f1_score": "--",
            "confusion_matrix": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
            "training_history": {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []},
            "last_training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        save_app_document("training_metrics", training_metrics)
        
        logger.info(f"Model training started with {total_images} images")
        training_progress.update({
            "status": "training",
            "percentage": 1,
            "current_epoch": 0,
            "total_epochs": 0,
            "current_stage": "Initializing",
            "message": "Initializing model training...",
            "error": None
        })
        
        # Start actual training in background thread
        import threading
        training_thread = threading.Thread(
            target=_run_training_thread,
            args=(image_paths, image_group_ids, dataset_fingerprint, training_scope)
        )
        training_thread.daemon = True
        training_thread.start()
        lock_acquired = False
        
        return jsonify({
            "success": True,
            "message": (
                f"Model training started with {total_images} "
                f"{training_scope} image(s)"
            ),
            "total_images": total_images
        })
    except Exception as e:
        if lock_acquired:
            training_lock.release()
        logger.exception("Failed to start model training")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/admin/dataset-groups-template", methods=["GET"])
@login_required
def download_image_group_manifest_template():
    try:
        template_file = render_image_group_manifest_template()
    except ValueError as error:
        return jsonify({"success": False, "error": str(error)}), 400
    return send_file(
        template_file,
        mimetype="text/csv",
        as_attachment=True,
        download_name="image_groups.csv"
    )

def _run_training_thread(image_paths, image_group_ids, dataset_fingerprint, training_scope):
    try:
        train_model_background(
            image_paths,
            image_group_ids,
            dataset_fingerprint,
            training_scope,
        )
    finally:
        training_lock.release()


def train_model_background(image_paths, image_group_ids, dataset_fingerprint, training_scope):
    """Background function to train the model with real ML"""
    global training_progress
    training_ai = copy.copy(orchid_ai)
    try:
        if dataset_fingerprint != get_dataset_fingerprint(image_paths, image_group_ids):
            raise RuntimeError("Dataset files changed before training began; start training again.")

        training_progress["status"] = "training"
        training_progress["current_stage"] = "Initializing"
        training_progress["message"] = "Creating species model (ImageNet weights may need to load)..."
        training_progress["percentage"] = 5
        logger.info("Starting real ML model training...")
        
        models_dir = os.path.join(BASE_DIR, "models")
        metrics_file = os.path.join(models_dir, "training_metrics.json")
        
        # Create feature-based classifiers
        training_ai.species_model = training_ai.create_species_model()
        training_progress["percentage"] = 7
        training_progress["message"] = "Creating health model..."
        training_ai.health_model = training_ai.create_health_model()
        
        # Prepare training data manually due to complex folder structure
        training_progress["percentage"] = 8
        training_progress["message"] = "Loading and preparing species images..."
        species_all_data = []
        species_all_labels = []
        health_all_labels = []
        species_all_paths = []
        image_sharpness_scores = []
        selected_image_paths = set(image_paths)
        
        for species_idx, species in enumerate(training_ai.species_classes):
            species_path = os.path.join(DATASET_DIR, species)
            if os.path.exists(species_path):
                # Handle both folder naming conventions: "not healthy" and "not_healthy"
                for health in ['healthy', 'not healthy', 'not_healthy']:
                    health_path = os.path.join(species_path, health)
                    if os.path.exists(health_path):
                        images = sorted(glob.glob(os.path.join(health_path, '*.*')))
                        images = [f for f in images if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
                        logger.info(f"Found {len(images)} images for {species}/{health}")
                        
                        for img_path in images:
                            relative_image_path = os.path.relpath(
                                img_path,
                                DATASET_DIR,
                            ).replace(os.sep, "/")
                            if relative_image_path not in selected_image_paths:
                                continue
                            try:
                                with Image.open(img_path) as source_image:
                                    img = ImageOps.exif_transpose(source_image).convert('RGB')
                                    img = img.resize(training_ai.img_size)
                                img_array = np.array(img, dtype=np.float32) / 255.0
                                img_array = np.clip(img_array, 0, 1)  # Ensure values are in valid range
                                species_all_data.append(img_array)
                                species_all_labels.append(species_idx)
                                health_all_labels.append(0 if health == 'healthy' else 1)
                                species_all_paths.append(relative_image_path)
                                image_sharpness_scores.append(
                                    ValidationCalibratedDetector.image_sharpness(
                                        np.asarray(img)
                                    )
                                )
                            except Exception as e:
                                logger.warning(f"Failed to load image {img_path}: {e}")
        
        if not species_all_data:
            raise RuntimeError("No species images were available for model training")

        species_all_data = np.asarray(species_all_data, dtype=np.float32)
        species_all_labels = np.asarray(species_all_labels, dtype=np.int64)
        health_all_labels = np.asarray(health_all_labels, dtype=np.int64)
        species_group_ids = np.asarray(
            [image_group_ids[image_path] for image_path in species_all_paths]
        )

        training_progress["message"] = "Creating group-safe dataset splits..."
        training_progress["percentage"] = 12
        (
            species_train_indices,
            species_validation_indices,
            species_test_indices
        ) = make_grouped_train_validation_test_split(
            species_all_labels,
            species_group_ids,
            "species classification"
        )
        (
            health_train_indices,
            health_validation_indices,
            health_test_indices
        ) = make_grouped_train_validation_test_split(
            health_all_labels,
            species_group_ids,
            "health classification"
        )

        all_indices = np.arange(len(species_all_data))
        species_development_indices = np.sort(np.concatenate([
            species_train_indices,
            species_validation_indices
        ]))
        health_development_indices = np.sort(np.concatenate([
            health_train_indices,
            health_validation_indices
        ]))
        development_indices = np.setdiff1d(
            all_indices,
            np.union1d(species_test_indices, health_test_indices)
        )
        development_indices = np.intersect1d(
            development_indices,
            np.union1d(
                species_development_indices,
                health_development_indices
            )
        )
        training_progress["message"] = "Augmenting training and validation images..."
        training_progress["percentage"] = 15
        augmented_images = training_ai.augment_training_images(
            species_all_data[development_indices],
            copies=1
        )
        combined_images = np.concatenate([
            species_all_data,
            augmented_images
        ], axis=0)
        training_progress["message"] = "Extracting CNN features for train/validation images..."
        training_progress["percentage"] = 18
        combined_features = training_ai.extract_image_features(
            combined_images,
            progress_range=(18, 40)
        )
        image_count = len(species_all_data)
        all_features = combined_features[:image_count]
        augmented_features = combined_features[image_count:]
        development_positions = {
            index: position
            for position, index in enumerate(development_indices)
        }

        species_train_labels = species_all_labels[species_train_indices]
        species_validation_labels = species_all_labels[species_validation_indices]
        species_test_labels = species_all_labels[species_test_indices]
        species_train_base_features = all_features[species_train_indices]
        species_validation_features = all_features[species_validation_indices]
        species_test_features = all_features[species_test_indices]
        health_train_labels = health_all_labels[health_train_indices]
        health_validation_labels = health_all_labels[health_validation_indices]
        health_test_labels = health_all_labels[health_test_indices]
        health_train_base_features = all_features[health_train_indices]
        health_validation_features = all_features[health_validation_indices]
        health_test_features = all_features[health_test_indices]

        species_augmented_train_indices = np.asarray([
            index for index in species_train_indices
            if index in development_positions
        ], dtype=np.int64)
        species_train_features = np.concatenate([
            species_train_base_features,
            augmented_features[
                [development_positions[index] for index in species_augmented_train_indices]
            ]
        ], axis=0)
        species_train_labels = np.concatenate([
            species_train_labels,
            species_all_labels[species_augmented_train_indices]
        ])
        health_augmented_train_indices = np.asarray([
            index for index in health_train_indices
            if index in development_positions
        ], dtype=np.int64)
        health_train_features = np.concatenate([
            health_train_base_features,
            augmented_features[
                [development_positions[index] for index in health_augmented_train_indices]
            ]
        ], axis=0)
        health_train_labels = np.concatenate([
            health_train_labels,
            health_all_labels[health_augmented_train_indices]
        ])

        training_progress["message"] = "Preparing validation and training sets..."
        training_progress["percentage"] = 40

        training_progress["message"] = "Training species classifier..."
        training_progress["percentage"] = 40
        training_ai.species_model.fit(species_train_features, species_train_labels)
        species_train_accuracy = float(
            accuracy_score(
                species_all_labels[species_train_indices],
                training_ai.species_model.predict(species_train_base_features)
            )
        )
        species_train_loss = float(log_loss(
            species_all_labels[species_train_indices],
            training_ai.species_model.predict_proba(species_train_base_features),
            labels=np.arange(len(training_ai.species_classes))
        ))
        training_progress["percentage"] = 52

        training_progress["current_stage"] = "Health Model"
        training_progress["message"] = "Training health classifier..."
        training_ai.health_model.fit(health_train_features, health_train_labels)
        species_validation_probabilities = training_ai.species_model.predict_proba(
            species_validation_features
        )
        species_validation_predictions = np.argmax(species_validation_probabilities, axis=1)
        species_validation_accuracy = float(accuracy_score(
            species_validation_labels,
            species_validation_predictions
        ))
        species_validation_loss = float(log_loss(
            species_validation_labels,
            species_validation_probabilities,
            labels=np.arange(len(training_ai.species_classes))
        ))
        health_validation_probabilities = training_ai.health_model.predict_proba(
            health_validation_features
        )
        health_validation_predictions = np.argmax(health_validation_probabilities, axis=1)
        health_validation_accuracy = float(accuracy_score(
            health_validation_labels,
            health_validation_predictions
        ))

        training_progress["message"] = "Calibrating prediction uncertainty..."
        training_progress["percentage"] = 68
        detector = ValidationCalibratedDetector(alpha=0.1)
        detector.fit(
            train_features=species_train_base_features,
            train_labels=species_all_labels[species_train_indices],
            validation_features=species_validation_features,
            validation_labels={
                "species": species_validation_labels,
                "health": health_validation_labels,
            },
            validation_probabilities={
                "species": species_validation_probabilities,
                "health": health_validation_probabilities,
            },
            probability_classes={
                "species": training_ai.species_classes,
                "health": training_ai.health_classes,
            },
            validation_sharpness=np.asarray(image_sharpness_scores)[
                species_validation_indices
            ],
            classes=training_ai.species_classes,
        )
        training_ai.unknown_detector = detector

        species_test_probabilities = training_ai.species_model.predict_proba(species_test_features)
        species_test_predictions = np.argmax(species_test_probabilities, axis=1)
        species_test_accuracy = float(accuracy_score(species_test_labels, species_test_predictions))
        species_test_precision, species_test_recall, species_test_f1, _ = precision_recall_fscore_support(
            species_test_labels,
            species_test_predictions,
            average='macro',
            zero_division=0
        )
        metrics = load_app_document("training_metrics", metrics_file, {})
        species_test_loss = float(log_loss(
            species_test_labels,
            species_test_probabilities,
            labels=np.arange(len(training_ai.species_classes))
        ))
        metrics["training_history"] = {
            "accuracy": [species_train_accuracy],
            "loss": [species_train_loss],
            "val_accuracy": [species_validation_accuracy],
            "val_loss": [species_validation_loss]
        }
        metrics["species_confusion_matrix"] = confusion_matrix(
            species_test_labels,
            species_test_predictions,
            labels=np.arange(len(training_ai.species_classes))
        ).tolist()
        species_report = classification_report(
            species_test_labels,
            species_test_predictions,
            labels=np.arange(len(training_ai.species_classes)),
            target_names=training_ai.species_classes,
            output_dict=True,
            zero_division=0
        )
        metrics["species_classification_report"] = {
            species: species_report[species]
            for species in training_ai.species_classes
        }

        health_test_probabilities = training_ai.health_model.predict_proba(health_test_features)
        health_test_predictions = np.argmax(health_test_probabilities, axis=1)
        health_test_accuracy = float(accuracy_score(health_test_labels, health_test_predictions))
        health_matrix = confusion_matrix(health_test_labels, health_test_predictions, labels=[0, 1])
        metrics["confusion_matrix"] = {
            "tn": int(health_matrix[0, 0]),
            "fp": int(health_matrix[0, 1]),
            "fn": int(health_matrix[1, 0]),
            "tp": int(health_matrix[1, 1])
        }
        health_test_precision, health_test_recall, health_test_f1, _ = precision_recall_fscore_support(
            health_test_labels,
            health_test_predictions,
            average='macro',
            zero_division=0
        )
        metrics["health_accuracy"] = round(health_test_accuracy, 4)
        metrics["health_precision"] = round(float(health_test_precision), 4)
        metrics["health_recall"] = round(float(health_test_recall), 4)
        metrics["health_f1_score"] = round(float(health_test_f1), 4)
        health_report = classification_report(
            health_test_labels,
            health_test_predictions,
            labels=[0, 1],
            target_names=training_ai.health_classes,
            output_dict=True,
            zero_division=0
        )
        metrics["health_classification_report"] = {
            label: health_report[label]
            for label in training_ai.health_classes
        }
        metrics["evaluation_split"] = {
            "method": "stratified_group_train_validation_test",
            "group_source": "datasets/image_groups.csv",
            "group_manifest_verified": True,
            "test_set_used_for_final_metrics_only": True,
            "species": {
                "train_images": int(len(species_train_indices)),
                "validation_images": int(len(species_validation_indices)),
                "test_images": int(len(species_test_indices)),
                "train_groups": int(len(np.unique(species_group_ids[species_train_indices]))),
                "validation_groups": int(len(np.unique(species_group_ids[species_validation_indices]))),
                "test_groups": int(len(np.unique(species_group_ids[species_test_indices])))
            },
            "health": {
                "train_images": int(len(health_train_indices)),
                "validation_images": int(len(health_validation_indices)),
                "test_images": int(len(health_test_indices)),
                "train_groups": int(len(np.unique(species_group_ids[health_train_indices]))),
                "validation_groups": int(len(np.unique(species_group_ids[health_validation_indices]))),
                "test_groups": int(len(np.unique(species_group_ids[health_test_indices])))
            },
            "validation_accuracy": {
                "species": round(species_validation_accuracy, 4),
                "health": round(health_validation_accuracy, 4)
            },
            "test_loss": {"species": round(species_test_loss, 4)}
        }
        metrics["unknown_detection"] = {
            "method": (
                "pooled split-conformal prediction sets and standardized CNN-feature "
                "distance thresholds calibrated on held-out validation groups"
            ),
            "alpha": detector.alpha,
            "distance_calibration_counts": detector.distance_calibration_counts,
            "distance_thresholds": detector.distance_thresholds,
            "validation_false_rejection_rate": "not estimated; no labeled out-of-distribution validation set",
        }

        if dataset_fingerprint != get_dataset_fingerprint(image_paths, image_group_ids):
            raise RuntimeError("Dataset files changed during training; evaluation results were discarded.")

        training_progress["message"] = "Training final models and saving..."
        training_progress["percentage"] = 90
        training_ai.species_feature_extractor.save(
            os.path.join(models_dir, "species_feature_extractor_224.keras")
        )
        joblib.dump(training_ai.species_model, os.path.join(models_dir, "species_classifier_224.joblib"))
        joblib.dump(training_ai.health_model, os.path.join(models_dir, "health_classifier_224.joblib"))
        detector_path = os.path.join(models_dir, "unknown_detection_224.json")
        with open(detector_path, "w", encoding="utf-8") as detector_file:
            json.dump(detector.to_dict(), detector_file)

        metrics["status"] = "completed"
        metrics["accuracy"] = round(species_test_accuracy, 4)
        metrics["precision"] = round(float(species_test_precision), 4)
        metrics["recall"] = round(float(species_test_recall), 4)
        metrics["f1_score"] = round(float(species_test_f1), 4)
        metrics["end_time"] = datetime.now().isoformat()
        metrics["last_training_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["using_real_model"] = True
        metrics["training_scope"] = training_scope
        metrics["dataset_fingerprint"] = dataset_fingerprint
        metrics["group_manifest_verified"] = True
        metrics.pop("error", None)
        save_app_document("training_metrics", metrics)

        orchid_ai.species_feature_extractor = training_ai.species_feature_extractor
        orchid_ai.species_model = training_ai.species_model
        orchid_ai.health_model = training_ai.health_model
        orchid_ai.unknown_detector = training_ai.unknown_detector
        orchid_ai.img_size = training_ai.img_size
        orchid_ai.use_real_model = True
        orchid_ai._model_artifact_signature = None
        training_progress["percentage"] = 100
        training_progress["status"] = "completed"
        training_progress["current_stage"] = "Complete"
        training_progress["message"] = "Training completed successfully!"
        logger.info(
            "Feature-based training completed. Group-safe species test accuracy: %.4f; health test accuracy: %.4f",
            species_test_accuracy,
            health_test_accuracy
        )
        
    except Exception as e:
        logger.exception("Model training failed")
        training_progress.update({
            "status": "failed",
            "message": f"Training failed: {e}",
            "error": str(e)
        })
        try:
            models_dir = os.path.join(BASE_DIR, "models")
            metrics_file = os.path.join(models_dir, "training_metrics.json")
            metrics = load_app_document("training_metrics", metrics_file, {})
            metrics["status"] = "failed"
            metrics["error"] = str(e)
            metrics["end_time"] = datetime.now().isoformat()
            save_app_document("training_metrics", metrics)
        except Exception:
            logger.exception("Failed to record model training failure")

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        # Check if file upload or JSON data
        if 'file' in request.files:
            # Handle file upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({"success": False, "error": "No file selected"})
            
            # Read file and convert to base64
            file_data = file.read()
            import base64
            image_data = f"data:image/jpeg;base64,{base64.b64encode(file_data).decode('utf-8')}"
        else:
            # Handle JSON data
            data = request.get_json(force=True)
            image_data = data.get("image")

        if not image_data:
            return jsonify({"success": False, "error": "No image provided."})

        result = orchid_ai.analyze_image(image_data)

        if result["success"]:
            filename = f"analysis_{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            if image_data.startswith("data:image"):
                _, encoded = image_data.split(",", 1)
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(encoded))
                result["image_url"] = f"/static/uploads/{filename}"

            # Save scan record
            scan_record = {
                "species": result.get("species"),
                "health": result.get("health"),
                "species_confidence": result.get("species_confidence", 0),
                "health_confidence": result.get("health_confidence", 0),
                "image_url": result.get("image_url", ""),
                "user_agent": request.headers.get('User-Agent', 'Unknown')
            }
            save_scan_record(scan_record)

            return jsonify(result)
        
        return jsonify(result)
    except Exception as e:
        logger.exception("Analysis failed")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/analyze-upload", methods=["POST"])
def api_analyze_upload():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image uploaded."})

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No image selected."})

        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Invalid file type. Use PNG, JPG, or JPEG."})

        if not verify_image_file(file):
            return jsonify({"success": False, "error": "Invalid image file."})

        filename = f"upload_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        image = Image.open(file).convert("RGB")
        image.save(filepath, format="JPEG", quality=90)

        result = orchid_ai.analyze_image(filepath)
        
        if result["success"]:
            result["image_url"] = f"/static/uploads/{filename}"
            return jsonify(result)
        
        return jsonify(result)
    except Exception as e:
        logger.exception("Upload analysis failed")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/preprocess/background-removal", methods=["POST"])
def api_background_removal_preview():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image uploaded."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No image selected."}), 400

        if not allowed_file(file.filename) or not verify_image_file(file):
            return jsonify({"success": False, "error": "Invalid image file."}), 400

        image = Image.open(file).convert("RGB")
        foreground = orchid_ai.remove_background(image)
        buffered = BytesIO()
        foreground.save(buffered, format="JPEG", quality=90)
        preview_image = base64.b64encode(buffered.getvalue()).decode("ascii")
        return jsonify({
            "success": True,
            "preview_image": f"data:image/jpeg;base64,{preview_image}"
        })
    except Exception as e:
        logger.exception("Background-removal preview failed")
        return jsonify({"success": False, "error": str(e)}), 500

# =========================================================
# STATIC FILES
# =========================================================
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/api/admin/training-progress")
def api_training_progress():
    """Get current training progress"""
    return jsonify(training_progress)

@app.route("/static/dataset/<path:filename>")
def dataset_file(filename):
    return send_from_directory(app.config["DATASET_FOLDER"], filename)

@app.route("/datasets/<path:filename>")
def serve_dataset(filename):
    """Serve dataset images from the datasets folder"""
    return send_from_directory(DATASET_DIR, filename)

# =========================================================
# ERROR HANDLERS
# =========================================================
@app.errorhandler(404)
def not_found(error):
    return render_template("404.html", app_name=APP_NAME), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template("500.html", app_name=APP_NAME), 500

# =========================================================
# STARTUP
# =========================================================
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    if orchid_ai.load_saved_models():
        print("✅ Loaded trained image models")
    else:
        print("ℹ️ Trained model files are not available; image scoring will be unavailable")
    
    print("=" * 60)
    print(f"🌸 {APP_NAME}")
    print("=" * 60)
    print("🌐 Server: https://localhost:5000")
    print("📱 Mobile: https://10.35.31.240:5000")
    print("⚠️  Accept the security warning in your browser")
    
    import ssl
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain('cert.pem', 'key.pem')
    
    # Debug mode restarts the app twice and adds a lot of startup lag.
    # For local production use, disable the reloader and keep the HTTPS config.
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=5000, threaded=True, ssl_context=context)