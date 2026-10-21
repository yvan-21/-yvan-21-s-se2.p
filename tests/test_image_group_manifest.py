import base64
import csv
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub heavy ML dependencies so the app can import in a minimal test environment.
for module_name in [
    "tensorflow",
    "keras",
    "skimage",
    "skimage.feature",
    "sklearn",
    "sklearn.model_selection",
    "sklearn.metrics",
    "sklearn.linear_model",
    "sklearn.pipeline",
    "sklearn.preprocessing",
    "joblib",
]:
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)

sys.modules["tensorflow"].get_logger = lambda: types.SimpleNamespace(setLevel=lambda *args, **kwargs: None)
class DummyCallback:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return None

keras_callbacks = types.ModuleType("tensorflow.keras.callbacks")
keras_callbacks.Callback = DummyCallback
sys.modules["tensorflow"].keras = types.SimpleNamespace(callbacks=keras_callbacks)
sys.modules["keras"].layers = types.SimpleNamespace()
sys.modules["keras"].models = types.SimpleNamespace()
sys.modules["sklearn.model_selection"].StratifiedGroupKFold = type("Dummy", (), {})
sys.modules["sklearn.metrics"].accuracy_score = lambda *args, **kwargs: 0.0
sys.modules["sklearn.metrics"].classification_report = lambda *args, **kwargs: ""
sys.modules["sklearn.metrics"].confusion_matrix = lambda *args, **kwargs: None
sys.modules["sklearn.metrics"].log_loss = lambda *args, **kwargs: 0.0
sys.modules["sklearn.metrics"].precision_recall_fscore_support = lambda *args, **kwargs: ([], [], [], [])
sys.modules["sklearn.linear_model"].LogisticRegression = type("Dummy", (), {})
sys.modules["sklearn.pipeline"].make_pipeline = lambda *args, **kwargs: None
sys.modules["sklearn.preprocessing"].StandardScaler = type("Dummy", (), {})
sys.modules["joblib"].dump = lambda *args, **kwargs: None
sys.modules["joblib"].load = lambda *args, **kwargs: None
sys.modules["skimage.feature"].graycomatrix = lambda *args, **kwargs: None
sys.modules["skimage.feature"].graycoprops = lambda *args, **kwargs: None

import app


class ImageGroupManifestRepairTest(unittest.TestCase):
    def test_health_guidance_explains_model_result_and_visible_cues(self):
        guidance = app.OrchidAI().get_detailed_recommendations(
            "Vanda",
            "not_healthy",
            93.0,
            76.3,
            symptom_observations={
                "yellow_percent": 4.2,
                "brown_dark_percent": 2.5,
            },
            species_tentative=True,
        )

        self.assertNotIn("Health result:", guidance)
        self.assertNotIn("health model flagged this image", guidance)
        self.assertNotIn("Image measurements:", guidance)
        self.assertNotIn("Color screening detected", guidance)
        self.assertIn("What to do\n1. Not-healthy result:", guidance)
        self.assertIn("Yellowing detected:", guidance)
        self.assertIn("Dark or brown areas detected:", guidance)
        self.assertNotIn("Species identification is uncertain", guidance)

    def test_healthy_model_result_is_not_repeated_as_guidance(self):
        guidance = app.OrchidAI().get_detailed_recommendations(
            "Vanda",
            "healthy",
            93.0,
            76.3,
            {"yellow_percent": 8.0, "brown_dark_percent": 4.0},
        )

        self.assertNotIn("Health result:", guidance)
        self.assertIn("What to do\n1. Healthy result:", guidance)
        self.assertIn("bright, indirect light", guidance)
        self.assertIn("Check the upper and lower leaf surfaces", guidance)
        self.assertNotIn("Image measurements:", guidance)
        self.assertNotIn("Yellowing detected:", guidance)
        self.assertNotIn("Dark or brown areas detected:", guidance)

    def test_failed_retraining_keeps_the_active_models_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_base_dir = app.BASE_DIR
            original_dataset_dir = app.DATASET_DIR
            original_species_classes = app.orchid_ai.species_classes
            original_models = (
                app.orchid_ai.species_model,
                app.orchid_ai.health_model,
                app.orchid_ai.species_feature_extractor,
                app.orchid_ai._model_artifact_signature,
                app.orchid_ai.use_real_model,
            )
            original_progress = dict(app.training_progress)
            active_species_model = object()
            active_health_model = object()
            active_feature_extractor = object()
            app.BASE_DIR = temp_dir
            app.DATASET_DIR = os.path.join(temp_dir, "empty-dataset")
            app.orchid_ai.species_classes = ["Cattleya"]
            app.orchid_ai.species_model = active_species_model
            app.orchid_ai.health_model = active_health_model
            app.orchid_ai.species_feature_extractor = active_feature_extractor
            app.orchid_ai._model_artifact_signature = ("active-model",)
            app.orchid_ai.use_real_model = True
            try:
                with patch.object(
                    app,
                    "get_dataset_fingerprint",
                    return_value="fingerprint",
                ), patch.object(
                    app.orchid_ai,
                    "create_species_model",
                    return_value=object(),
                ), patch.object(
                    app.orchid_ai,
                    "create_health_model",
                    return_value=object(),
                ), patch.object(
                    app,
                    "load_app_document",
                    return_value={},
                ), patch.object(app, "save_app_document"):
                    app.train_model_background([], {}, "fingerprint", "all")
                    self.assertIs(app.orchid_ai.species_model, active_species_model)
                    self.assertIs(app.orchid_ai.health_model, active_health_model)
                    self.assertIs(
                        app.orchid_ai.species_feature_extractor,
                        active_feature_extractor,
                    )
                    self.assertEqual(
                        app.orchid_ai._model_artifact_signature,
                        ("active-model",),
                    )
                    self.assertTrue(app.orchid_ai.use_real_model)
            finally:
                app.BASE_DIR = original_base_dir
                app.DATASET_DIR = original_dataset_dir
                app.orchid_ai.species_classes = original_species_classes
                (
                    app.orchid_ai.species_model,
                    app.orchid_ai.health_model,
                    app.orchid_ai.species_feature_extractor,
                    app.orchid_ai._model_artifact_signature,
                    app.orchid_ai.use_real_model,
                ) = original_models
                app.training_progress.clear()
                app.training_progress.update(original_progress)

    def test_new_upload_registry_separates_old_new_and_all_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "dataset_uploads.json")
            original_registry_path = app.get_dataset_upload_registry_path
            original_get_image_paths = app.get_dataset_image_paths
            app.get_dataset_upload_registry_path = lambda: registry_path
            app.get_dataset_image_paths = lambda: [
                "Cattleya/healthy/old.jpg",
                "Cattleya/healthy/new.jpg",
            ]
            try:
                app.mark_dataset_image_as_new_upload("Cattleya/healthy/new.jpg")
                old_paths = app.get_training_image_paths("old")
                new_paths = app.get_training_image_paths("new")
                all_paths = app.get_training_image_paths("all")
            finally:
                app.get_dataset_upload_registry_path = original_registry_path
                app.get_dataset_image_paths = original_get_image_paths

        self.assertEqual(old_paths, ["Cattleya/healthy/old.jpg"])
        self.assertEqual(new_paths, ["Cattleya/healthy/new.jpg"])
        self.assertEqual(len(all_paths), 2)

    def test_uploaded_image_gets_automatic_plant_group_and_preserves_existing_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, "datasets")
            old_path = "Cattleya/healthy/old.jpg"
            new_path = "Cattleya/healthy/new.jpg"
            for relative_path in (old_path, new_path):
                file_path = os.path.join(dataset_dir, *relative_path.split("/"))
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as image_file:
                    image_file.write(relative_path.encode("utf-8"))
            manifest_path = os.path.join(dataset_dir, "image_groups.csv")
            with open(manifest_path, "w", newline="", encoding="utf-8") as manifest:
                writer = csv.DictWriter(
                    manifest,
                    fieldnames=["image_path", "plant_id", "session_id"],
                )
                writer.writeheader()
                writer.writerow({
                    "image_path": old_path,
                    "plant_id": "plant_existing",
                    "session_id": "",
                })
                writer.writerow({
                    "image_path": new_path,
                    "plant_id": "",
                    "session_id": "",
                })

            original_dataset_dir = app.DATASET_DIR
            original_manifest_path = app.get_image_group_manifest_path
            original_get_image_paths = app.get_dataset_image_paths
            app.DATASET_DIR = dataset_dir
            app.get_image_group_manifest_path = lambda: manifest_path
            app.get_dataset_image_paths = lambda: [old_path, new_path]
            try:
                app.register_uploaded_image_group(new_path)
                with open(manifest_path, newline="", encoding="utf-8-sig") as manifest:
                    rows = {
                        row["image_path"]: row
                        for row in csv.DictReader(manifest)
                    }
            finally:
                app.DATASET_DIR = original_dataset_dir
                app.get_image_group_manifest_path = original_manifest_path
                app.get_dataset_image_paths = original_get_image_paths

        self.assertEqual(rows[old_path]["plant_id"], "plant_existing")
        self.assertTrue(rows[new_path]["plant_id"].startswith("upload_"))
        self.assertEqual(rows[new_path]["session_id"], "")

    def test_selected_scope_can_load_its_groups_while_other_images_remain_in_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, "datasets")
            old_path = "Cattleya/healthy/old.jpg"
            new_path = "Cattleya/healthy/new.jpg"
            for relative_path in (old_path, new_path):
                file_path = os.path.join(dataset_dir, *relative_path.split("/"))
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as image_file:
                    image_file.write(relative_path.encode("utf-8"))

            manifest_path = os.path.join(dataset_dir, "image_groups.csv")
            with open(manifest_path, "w", newline="", encoding="utf-8") as manifest:
                writer = csv.DictWriter(
                    manifest,
                    fieldnames=["image_path", "plant_id", "session_id"],
                )
                writer.writeheader()
                writer.writerow({
                    "image_path": old_path,
                    "plant_id": "plant_old",
                    "session_id": "",
                })
                writer.writerow({
                    "image_path": new_path,
                    "plant_id": "plant_new",
                    "session_id": "",
                })

            original_dataset_dir = app.DATASET_DIR
            original_manifest_path = app.get_image_group_manifest_path
            original_get_image_paths = app.get_dataset_image_paths
            app.DATASET_DIR = dataset_dir
            app.get_image_group_manifest_path = lambda: manifest_path
            app.get_dataset_image_paths = lambda: [old_path, new_path]
            try:
                groups, error = app.load_image_group_assignments(
                    [new_path],
                    allow_extra_paths=True,
                )
            finally:
                app.DATASET_DIR = original_dataset_dir
                app.get_image_group_manifest_path = original_manifest_path
                app.get_dataset_image_paths = original_get_image_paths

        self.assertIsNone(error)
        self.assertEqual(set(groups), {new_path})

    def test_dashboard_upload_is_tracked_as_new_and_added_to_group_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, "datasets")
            old_path = "Cattleya/healthy/old.jpg"
            old_file = os.path.join(dataset_dir, *old_path.split("/"))
            os.makedirs(os.path.dirname(old_file), exist_ok=True)
            with open(old_file, "wb") as image_file:
                image_file.write(b"old image")
            manifest_path = os.path.join(dataset_dir, "image_groups.csv")
            with open(manifest_path, "w", newline="", encoding="utf-8") as manifest:
                writer = csv.DictWriter(
                    manifest,
                    fieldnames=["image_path", "plant_id", "session_id"],
                )
                writer.writeheader()
                writer.writerow({
                    "image_path": old_path,
                    "plant_id": "plant_existing",
                    "session_id": "",
                })

            original_base_dir = app.BASE_DIR
            original_dataset_dir = app.DATASET_DIR
            app.BASE_DIR = temp_dir
            app.DATASET_DIR = dataset_dir
            client = app.app.test_client()
            with client.session_transaction() as login_session:
                login_session["logged_in"] = True
                login_session["user_type"] = "admin"
            try:
                response = client.post(
                    "/api/admin/upload-dataset",
                    data={
                        "species": "Cattleya",
                        "health": "healthy",
                        "file": (io.BytesIO(b"new image"), "new.jpg"),
                    },
                    content_type="multipart/form-data",
                )
                with open(manifest_path, newline="", encoding="utf-8-sig") as manifest:
                    manifest_rows = {
                        row["image_path"]: row
                        for row in csv.DictReader(manifest)
                    }
                with open(
                    os.path.join(temp_dir, "models", "dataset_uploads.json"),
                    encoding="utf-8",
                ) as registry_file:
                    registry = json.load(registry_file)
            finally:
                app.BASE_DIR = original_base_dir
                app.DATASET_DIR = original_dataset_dir

        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["image_path"], "Cattleya/healthy/new.jpg")
        self.assertEqual(manifest_rows[old_path]["plant_id"], "plant_existing")
        self.assertTrue(manifest_rows[payload["image_path"]]["plant_id"].startswith("upload_"))
        self.assertEqual(
            registry["new_upload_paths"],
            ["Cattleya/healthy/new.jpg"],
        )

    def test_train_endpoint_uses_selected_scope_and_rejects_empty_scope(self):
        client = app.app.test_client()
        with client.session_transaction() as login_session:
            login_session["logged_in"] = True
            login_session["user_type"] = "admin"

        with patch.object(
            app,
            "get_dataset_image_paths",
            return_value=["Cattleya/healthy/old.jpg"],
        ), patch.object(app, "get_training_image_paths", return_value=[]):
            response = client.post(
                "/api/admin/train-model",
                json={"scope": "new"},
            )

        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertIn("new uploads has only 0 images", payload["error"])

    def test_admin_dashboard_exposes_training_scope_choices_in_both_sections(self):
        client = app.app.test_client()
        with client.session_transaction() as login_session:
            login_session["logged_in"] = True
            login_session["user_type"] = "admin"

        response = client.get("/admin/dashboard")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Existing images only", response.data)
        self.assertIn(b"Newly uploaded dataset images only", response.data)
        self.assertIn(b"All images (old + new)", response.data)
        self.assertNotIn(b"Model scores:", response.data)
        self.assertNotIn(b"Prediction Result", response.data)
        self.assertIn(b"centered square crop", response.data)
        self.assertIn(b'id="prediction-warning"', response.data)
        self.assertIn(b"MobileNetV2 maps that to", response.data)
        self.assertIn(b"Random example after this layer and all earlier augmentation layers", response.data)
        self.assertIn(b"GrabCut estimates foreground from an initial rectangle", response.data)
        self.assertIn(b"result.warning", response.data)
        self.assertNotIn(b"Likely ${result.health_prediction", response.data)

    def test_user_dashboard_renders_mobile_navigation_and_analysis_guidance(self):
        client = app.app.test_client()
        with client.session_transaction() as login_session:
            login_session["logged_in"] = True
            login_session["user_type"] = "user"

        response = client.get("/user/dashboard")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'class="mobile-nav-item" data-section="scan"', response.data)
        self.assertIn(b'class="mobile-nav-item" data-section="camera"', response.data)
        self.assertIn(b'id="result-recommendation-card"', response.data)
        self.assertIn(b'id="result-recommendation"', response.data)
        self.assertIn(b"Visual Screening &amp; Care Guidance", response.data)
        self.assertNotIn(b'id="result-scores"', response.data)
        self.assertNotIn(b'>Dataset Model Scores<', response.data)
        self.assertNotIn(b'Model Scores', response.data)
        self.assertNotIn(b'id="result-confidence"', response.data)
        self.assertIn(b'id="result-warning"', response.data)
        self.assertIn(
            b'Dataset-trained model scores:',
            response.data,
        )
        self.assertNotIn(b'Dataset-trained model scores|Image measurements', response.data)
        self.assertIn(b"function displayAnalysisResult(data, imageData)", response.data)
        self.assertNotIn(b"Likely ${data.health_prediction", response.data)
        self.assertIn(b"function renderCareGuidance(container, guidanceText)", response.data)
        self.assertIn(b'className = \'guidance-actions\'', response.data)
        self.assertIn(b"actionsTitle.textContent = 'What to do'", response.data)

    def test_mobile_prediction_pages_render_uncertainty_without_color_measurements(self):
        client = app.app.test_client()

        for route in ("/camera", "/upload"):
            with self.subTest(route=route):
                response = client.get(route)

                self.assertEqual(response.status_code, 200)
                self.assertIn(b"Unknown / uncertain", response.data)
                self.assertIn(b"data.warning", response.data)
                self.assertNotIn(b"LIKELY ${data.health_prediction", response.data)
                self.assertNotIn(b"Validation-calibrated model scores (not accuracy)", response.data)
                self.assertIn(b"Why this result & Care Guidance", response.data)
                self.assertNotIn(b"not calibrated", response.data)
                self.assertNotIn(b"Color Index", response.data)
                self.assertNotIn(b"data.color_index", response.data)

    def test_image_processing_page_handles_uncertain_health_without_model_scores(self):
        client = app.app.test_client()
        with client.session_transaction() as login_session:
            login_session["logged_in"] = True
            login_session["user_type"] = "admin"

        response = client.get("/image-processing")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"id=\"resultWarning\"", response.data)
        self.assertIn(b"UNKNOWN / UNCERTAIN", response.data)
        self.assertNotIn(b"LIKELY", response.data)
        self.assertNotIn(b"Calibrated scores unavailable; retrain the model", response.data)
        self.assertNotIn(b"not calibrated", response.data)

    def test_prediction_without_calibration_shows_model_labels_as_provisional(self):
        class FixedClassifier:
            def __init__(self, probabilities):
                self.probabilities = probabilities

            def predict(self, images, verbose=0):
                return self.probabilities

        image = app.Image.new("RGB", (128, 128), color=(40, 150, 70))
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        image_buffer.seek(0)

        orchid_ai = app.OrchidAI()
        orchid_ai.species_model = FixedClassifier(
            [[1.0 if index == 0 else 0.0 for index in range(len(orchid_ai.species_classes))]]
        )
        orchid_ai.health_model = FixedClassifier([[0.846, 0.154]])
        orchid_ai.species_feature_extractor = None
        orchid_ai.unknown_detector = None

        with patch.object(orchid_ai, "remove_background", side_effect=lambda image: image), \
             patch.object(orchid_ai, "extract_texture_features", return_value={}), \
             patch.object(orchid_ai, "extract_shape_features", return_value={}), \
             patch.object(orchid_ai, "analyze_visible_symptoms", return_value={}), \
             patch.object(orchid_ai, "get_detailed_recommendations", return_value="Care guidance"), \
             patch.object(orchid_ai, "get_training_augmentation_previews", return_value=([], None)):
            result = orchid_ai.predict_with_model(image_buffer)

        self.assertTrue(result["success"])
        self.assertEqual(result["species"], orchid_ai.species_classes[0])
        self.assertEqual(result["health_status"], "healthy")
        self.assertFalse(result["uncertainty_calibrated"])
        self.assertFalse(result["health_score_calibrated"])
        self.assertIn("unsupported-image detection is unavailable", result["warning"])
        self.assertIn("Provisional model prediction", result["warning"])

    def test_prediction_keeps_top_species_when_calibration_disagrees(self):
        class FixedClassifier:
            def __init__(self, probabilities):
                self.probabilities = probabilities

            def predict_proba(self, features):
                return self.probabilities

        class FixedDetector:
            def image_sharpness(self, image):
                return 100.0

            def calibrate_probabilities(self, probabilities, target):
                return probabilities

            def has_probability_calibration(self, target):
                return True

            def calibrate_probabilities(self, probabilities, target):
                return probabilities

            def has_probability_calibration(self, target):
                return True

            def prediction_set(self, probabilities, target):
                if target == "species":
                    return ["Phalaenopsis"]
                return ["healthy", "not_healthy"]

            def is_out_of_distribution(self, features, class_name):
                return True, 1.43

            def is_too_blurry(self, sharpness):
                return False

        image = app.Image.new("RGB", (128, 128), color=(40, 150, 70))
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        image_buffer.seek(0)

        orchid_ai = app.OrchidAI()
        orchid_ai.species_classes = [
            "Cattleya", "Dendrobium", "Phalaenopsis", "Vanda"
        ]
        orchid_ai.species_model = FixedClassifier([[0.01, 0.01, 0.05, 0.93]])
        orchid_ai.health_model = FixedClassifier([[0.846, 0.154]])
        orchid_ai.species_feature_extractor = object()
        orchid_ai.unknown_detector = FixedDetector()
        applications = types.SimpleNamespace(
            mobilenet_v2=types.SimpleNamespace(preprocess_input=lambda values: values)
        )

        with patch.object(orchid_ai, "remove_background", side_effect=lambda image: image), \
             patch.object(orchid_ai, "extract_image_features", return_value=[[0.0]]), \
             patch.object(orchid_ai, "extract_texture_features", return_value={}), \
             patch.object(orchid_ai, "extract_shape_features", return_value={}), \
             patch.object(orchid_ai, "analyze_visible_symptoms", return_value={}), \
             patch.object(orchid_ai, "get_detailed_recommendations", return_value="Care guidance"), \
             patch.object(orchid_ai, "get_training_augmentation_previews", return_value=([], None)), \
             patch.object(app.tf.keras, "applications", applications, create=True):
            result = orchid_ai.predict_with_model(image_buffer)

        self.assertTrue(result["success"])
        self.assertEqual(result["species"], "Vanda")
        self.assertTrue(result["out_of_distribution"])
        self.assertEqual(result["health_status"], "healthy")
        self.assertEqual(result["health_prediction"], "healthy")
        self.assertTrue(result["is_healthy"])
        self.assertTrue(result["health_score_calibrated"])
        self.assertNotIn("Likely Vanda", result["warning"])
        self.assertNotIn("calibrated candidates", result["warning"])
        self.assertTrue(result["unknown_or_uncertain"])

    def test_prediction_averages_full_frame_and_square_crop_for_species(self):
        class FixedClassifier:
            def __init__(self, probabilities):
                self.probabilities = probabilities

            def predict_proba(self, features):
                return self.probabilities[:len(features)]

        class FixedDetector:
            def image_sharpness(self, image):
                return 100.0

            def calibrate_probabilities(self, probabilities, target):
                return probabilities

            def has_probability_calibration(self, target):
                return True

            def prediction_set(self, probabilities, target):
                if target == "species":
                    return ["Cattleya", "Vanda"]
                return ["healthy", "not_healthy"]

            def is_out_of_distribution(self, features, class_name):
                return True, 1.43

            def is_too_blurry(self, sharpness):
                return False

        image = app.Image.new("RGB", (256, 128), color=(40, 150, 70))
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        image_buffer.seek(0)

        orchid_ai = app.OrchidAI()
        orchid_ai.species_classes = [
            "Cattleya", "Dendrobium", "Phalaenopsis", "Vanda"
        ]
        orchid_ai.species_model = FixedClassifier([
            [0.68, 0.0, 0.0, 0.32],
            [0.01, 0.0, 0.0, 0.99],
        ])
        orchid_ai.health_model = FixedClassifier([[0.846, 0.154]])
        orchid_ai.species_feature_extractor = object()
        orchid_ai.unknown_detector = FixedDetector()
        applications = types.SimpleNamespace(
            mobilenet_v2=types.SimpleNamespace(preprocess_input=lambda values: values)
        )

        with patch.object(orchid_ai, "remove_background", side_effect=lambda image: image), \
             patch.object(orchid_ai, "extract_image_features", return_value=[[0.0], [1.0]]) as extract_features, \
             patch.object(orchid_ai, "extract_texture_features", return_value={}), \
             patch.object(orchid_ai, "extract_shape_features", return_value={}), \
             patch.object(orchid_ai, "analyze_visible_symptoms", return_value={}), \
             patch.object(orchid_ai, "get_detailed_recommendations", return_value="Care guidance"), \
             patch.object(orchid_ai, "get_training_augmentation_previews", return_value=([], None)), \
             patch.object(app.tf.keras, "applications", applications, create=True):
            result = orchid_ai.predict_with_model(image_buffer)

        self.assertTrue(result["success"])
        self.assertEqual(extract_features.call_args.args[0].shape, (2, 224, 224, 3))
        self.assertEqual(result["species"], "Vanda")
        self.assertEqual(result["species_view_count"], 2)
        self.assertTrue(result["species_crop_image"].startswith("data:image/png;base64,"))
        self.assertNotIn("Likely Vanda", result["warning"])
        self.assertNotIn("calibrated candidates", result["warning"])
        self.assertTrue(result["unknown_or_uncertain"])
        guidance = orchid_ai.get_detailed_recommendations(
            result["species"],
            result["health_status"],
            result["species_confidence"],
            result["health_confidence"],
            species_tentative=True,
        )
        self.assertNotIn("Health result:", guidance)
        self.assertIn("bright, indirect light", guidance)
        self.assertNotIn("Species identification is uncertain", guidance)
        self.assertNotIn("calibrated candidates", guidance)

    def test_user_analysis_lazy_loads_saved_dataset_classifiers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = os.path.join(temp_dir, "models")
            os.makedirs(models_dir)
            artifact_names = [
                "species_feature_extractor_224.keras",
                "species_classifier_224.joblib",
                "health_classifier_224.joblib",
            ]
            for artifact_name in artifact_names:
                with open(os.path.join(models_dir, artifact_name), "wb") as artifact:
                    artifact.write(b"test artifact")

            original_base_dir = app.BASE_DIR
            original_models = (
                app.orchid_ai.species_model,
                app.orchid_ai.health_model,
                app.orchid_ai.species_feature_extractor,
                app.orchid_ai._model_artifact_signature,
                app.orchid_ai.use_real_model,
            )
            app.BASE_DIR = temp_dir
            app.orchid_ai.species_model = None
            app.orchid_ai.health_model = None
            app.orchid_ai.species_feature_extractor = None
            app.orchid_ai._model_artifact_signature = None
            app.orchid_ai.use_real_model = False
            extractor = types.SimpleNamespace(input_shape=(None, 224, 224, 3))
            classifier = object()
            expected = {"success": True, "species": "Cattleya"}
            try:
                with patch.object(
                    app.keras.models,
                    "load_model",
                    create=True,
                    return_value=extractor,
                ) as load_feature_extractor, patch.object(
                    app.joblib,
                    "load",
                    return_value=classifier,
                ) as load_classifier, patch.object(
                    app.orchid_ai,
                    "predict_with_model",
                    return_value=expected,
                ) as predict_with_model:
                    result = app.orchid_ai.analyze_image("image-data")
            finally:
                app.BASE_DIR = original_base_dir
                (
                    app.orchid_ai.species_model,
                    app.orchid_ai.health_model,
                    app.orchid_ai.species_feature_extractor,
                    app.orchid_ai._model_artifact_signature,
                    app.orchid_ai.use_real_model,
                ) = original_models

        self.assertEqual(result, expected)
        load_feature_extractor.assert_called_once()
        self.assertEqual(load_classifier.call_count, 2)
        predict_with_model.assert_called_once_with("image-data")

    def test_user_analysis_does_not_return_a_color_guess_without_model_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_base_dir = app.BASE_DIR
            original_models = (
                app.orchid_ai.species_model,
                app.orchid_ai.health_model,
                app.orchid_ai.species_feature_extractor,
                app.orchid_ai._model_artifact_signature,
                app.orchid_ai.use_real_model,
            )
            app.BASE_DIR = temp_dir
            app.orchid_ai.species_model = None
            app.orchid_ai.health_model = None
            app.orchid_ai.species_feature_extractor = None
            app.orchid_ai._model_artifact_signature = None
            app.orchid_ai.use_real_model = False
            try:
                with patch.object(
                    app.orchid_ai,
                    "fallback_analysis",
                    side_effect=AssertionError("Color fallback must not create a prediction"),
                ):
                    result = app.orchid_ai.analyze_image("image-data")
            finally:
                app.BASE_DIR = original_base_dir
                (
                    app.orchid_ai.species_model,
                    app.orchid_ai.health_model,
                    app.orchid_ai.species_feature_extractor,
                    app.orchid_ai._model_artifact_signature,
                    app.orchid_ai.use_real_model,
                ) = original_models

        self.assertFalse(result["success"])
        self.assertIn("Dataset-based analysis is temporarily unavailable", result["error"])

    def test_mobile_analyzer_calls_the_server_instead_of_using_color_guesses(self):
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "static",
            "js",
            "image_analysis.js",
        )
        with open(script_path, encoding="utf-8") as script_file:
            script = script_file.read()

        self.assertIn("fetch('/api/analyze'", script)
        self.assertNotIn("classifyByColor", script)
        self.assertNotIn("getRandomPrediction", script)
        for route in ("/upload", "/camera"):
            response = app.app.test_client().get(route)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn(
                b"Validation-calibrated model scores (not accuracy)",
                response.data,
            )
            self.assertIn(b"data.warning", response.data)
            self.assertNotIn(b"Color Index", response.data)

    def test_repair_assigns_group_ids_to_images_without_annotations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = os.path.join(temp_dir, "image_groups.csv")
            image_paths = [
                "Cattleya/healthy/a.jpg",
                "Cattleya/healthy/b.jpg",
                "Dendrobium/not healthy/c.jpg",
            ]

            with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_path", "plant_id", "session_id"])
                writer.writeheader()
                for path in image_paths:
                    writer.writerow({"image_path": path, "plant_id": "", "session_id": ""})

            original_manifest_path = app.get_image_group_manifest_path
            app.get_image_group_manifest_path = lambda: manifest_path
            try:
                repaired = app.repair_missing_group_manifest_ids(image_paths)
                rows, error = app.load_image_group_assignments(image_paths)
                self.assertTrue(repaired)
                self.assertIsNone(error)
                self.assertEqual(set(rows), set(image_paths))
                self.assertTrue(all(
                    rows[path].startswith("group-")
                    for path in image_paths
                ))

                with open(manifest_path, newline="", encoding="utf-8-sig") as handle:
                    manifest_rows = {
                        row["image_path"]: row
                        for row in csv.DictReader(handle)
                    }
                self.assertTrue(all(
                    manifest_rows[path]["plant_id"].startswith("upload_")
                    for path in image_paths
                ))
                self.assertEqual(
                    len({manifest_rows[path]["plant_id"] for path in image_paths}),
                    len(image_paths),
                )
            finally:
                app.get_image_group_manifest_path = original_manifest_path

    def test_train_endpoint_assigns_group_ids_before_starting_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, "datasets")
            image_paths = [
                f"Cattleya/healthy/image_{index}.jpg"
                for index in range(10)
            ]
            for index, image_path in enumerate(image_paths):
                image_file = os.path.join(dataset_dir, *image_path.split("/"))
                os.makedirs(os.path.dirname(image_file), exist_ok=True)
                with open(image_file, "wb") as handle:
                    handle.write(f"image-{index}".encode("utf-8"))

            manifest_path = os.path.join(dataset_dir, "image_groups.csv")
            with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["image_path", "plant_id", "session_id"],
                )
                writer.writeheader()
                for image_path in image_paths:
                    writer.writerow({
                        "image_path": image_path,
                        "plant_id": "",
                        "session_id": "",
                    })

            original_base_dir = app.BASE_DIR
            original_dataset_dir = app.DATASET_DIR
            app.BASE_DIR = temp_dir
            app.DATASET_DIR = dataset_dir
            client = app.app.test_client()
            with client.session_transaction() as login_session:
                login_session["logged_in"] = True
                login_session["user_type"] = "admin"
            try:
                with patch.object(app, "get_dataset_image_paths", return_value=image_paths), \
                     patch.object(app, "get_training_image_paths", return_value=image_paths), \
                     patch.object(
                         app.orchid_ai,
                         "get_available_species",
                         return_value=["Cattleya", "Dendrobium", "Phalaenopsis", "Vanda"],
                     ), \
                     patch.object(
                         app,
                         "get_dataset_labels_for_paths",
                         return_value=([], []),
                     ), \
                     patch.object(app, "make_grouped_train_validation_test_split"), \
                     patch.object(app, "save_app_document"), \
                     patch.object(app, "training_lock") as training_lock, \
                     patch("threading.Thread") as training_thread:
                    training_lock.acquire.return_value = True
                    response = client.post("/api/admin/train-model", json={"scope": "all"})

                with open(manifest_path, newline="", encoding="utf-8-sig") as handle:
                    manifest_rows = list(csv.DictReader(handle))
            finally:
                app.BASE_DIR = original_base_dir
                app.DATASET_DIR = original_dataset_dir

        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["total_images"], len(image_paths))
        self.assertTrue(all(row["plant_id"].startswith("upload_") for row in manifest_rows))
        self.assertEqual(
            len({row["plant_id"] for row in manifest_rows}),
            len(image_paths),
        )
        training_thread.return_value.start.assert_called_once()

    def test_legacy_generated_group_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = os.path.join(temp_dir, "image_groups.csv")
            image_paths = ["Cattleya/healthy/a.jpg", "Dendrobium/healthy/b.jpg"]
            with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_path", "plant_id", "session_id"])
                writer.writeheader()
                for index, path in ((1, image_paths[1]), (3, image_paths[0])):
                    writer.writerow({
                        "image_path": path,
                        "plant_id": f"plant_{index}",
                        "session_id": f"session_{index}",
                    })

            original_manifest_path = app.get_image_group_manifest_path
            app.get_image_group_manifest_path = lambda: manifest_path
            try:
                rows, error = app.load_image_group_assignments(image_paths)
            finally:
                app.get_image_group_manifest_path = original_manifest_path

        self.assertIsNone(rows)
        self.assertIn("automatically generated per-image IDs", error)

    def test_manifest_template_does_not_preserve_generated_placeholder_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = os.path.join(temp_dir, "image_groups.csv")
            image_paths = ["Cattleya/healthy/a.jpg", "Dendrobium/healthy/b.jpg"]
            with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_path", "plant_id", "session_id"])
                writer.writeheader()
                for index, path in ((1, image_paths[1]), (3, image_paths[0])):
                    writer.writerow({
                        "image_path": path,
                        "plant_id": f"plant_{index}",
                        "session_id": f"session_{index}",
                    })

            original_manifest_path = app.get_image_group_manifest_path
            original_get_image_paths = app.get_dataset_image_paths
            app.get_image_group_manifest_path = lambda: manifest_path
            app.get_dataset_image_paths = lambda: image_paths
            try:
                template = app.render_image_group_manifest_template()
                content = template.getvalue().decode("utf-8-sig")
            finally:
                app.get_image_group_manifest_path = original_manifest_path
                app.get_dataset_image_paths = original_get_image_paths

        self.assertIn("image_path,plant_id,session_id", content)
        self.assertIn("Cattleya/healthy/a.jpg,,", content)
        self.assertNotIn("plant_1", content)

    def test_identical_image_with_conflicting_species_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, "datasets")
            image_paths = [
                "Cattleya/healthy/a.jpg",
                "Dendrobium/healthy/b.jpg",
            ]
            for image_path in image_paths:
                image_file = os.path.join(dataset_dir, *image_path.split("/"))
                os.makedirs(os.path.dirname(image_file), exist_ok=True)
                with open(image_file, "wb") as handle:
                    handle.write(b"identical image bytes")

            manifest_path = os.path.join(dataset_dir, "image_groups.csv")
            with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_path", "plant_id", "session_id"])
                writer.writeheader()
                writer.writerow({
                    "image_path": image_paths[0],
                    "plant_id": "plant-a",
                    "session_id": "session-a",
                })
                writer.writerow({
                    "image_path": image_paths[1],
                    "plant_id": "plant-b",
                    "session_id": "session-b",
                })

            original_dataset_dir = app.DATASET_DIR
            original_manifest_path = app.get_image_group_manifest_path
            app.DATASET_DIR = dataset_dir
            app.get_image_group_manifest_path = lambda: manifest_path
            try:
                rows, error = app.load_image_group_assignments(image_paths)
            finally:
                app.DATASET_DIR = original_dataset_dir
                app.get_image_group_manifest_path = original_manifest_path

        self.assertIsNone(rows)
        self.assertIn("Identical image content has conflicting dataset labels", error)

    def test_identical_images_with_same_label_are_kept_in_one_group(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, "datasets")
            image_paths = [
                "Cattleya/healthy/a.jpg",
                "Cattleya/healthy/b.jpg",
            ]
            for image_path in image_paths:
                image_file = os.path.join(dataset_dir, *image_path.split("/"))
                os.makedirs(os.path.dirname(image_file), exist_ok=True)
                with open(image_file, "wb") as handle:
                    handle.write(b"same image")

            manifest_path = os.path.join(dataset_dir, "image_groups.csv")
            with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_path", "plant_id", "session_id"])
                writer.writeheader()
                for path, suffix in zip(image_paths, ("a", "b")):
                    writer.writerow({
                        "image_path": path,
                        "plant_id": f"plant-{suffix}",
                        "session_id": f"session-{suffix}",
                    })

            original_dataset_dir = app.DATASET_DIR
            original_manifest_path = app.get_image_group_manifest_path
            app.DATASET_DIR = dataset_dir
            app.get_image_group_manifest_path = lambda: manifest_path
            try:
                assignments, error = app.load_image_group_assignments(image_paths)
            finally:
                app.DATASET_DIR = original_dataset_dir
                app.get_image_group_manifest_path = original_manifest_path

        self.assertIsNone(error)
        self.assertEqual(assignments[image_paths[0]], assignments[image_paths[1]])

    def test_background_removal_preview_endpoint_returns_image(self):
        image_buffer = io.BytesIO()
        app.Image.new("RGB", (16, 16), color=(20, 140, 60)).save(image_buffer, format="PNG")
        image_buffer.seek(0)

        original_remove_background = app.orchid_ai.remove_background
        app.orchid_ai.remove_background = lambda image: image
        try:
            response = app.app.test_client().post(
                "/api/preprocess/background-removal",
                data={"image": (image_buffer, "leaf.png")},
                content_type="multipart/form-data",
            )
        finally:
            app.orchid_ai.remove_background = original_remove_background

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertTrue(payload["preview_image"].startswith("data:image/jpeg;base64,"))

    def test_care_guidance_omits_measurements_but_keeps_cue_based_advice(self):
        guidance = app.orchid_ai.get_detailed_recommendations(
            "Phalaenopsis",
            "not_healthy",
            82.5,
            74.0,
            {"yellow_percent": 8.0, "brown_dark_percent": 2.5},
        )

        self.assertNotIn("Health result:", guidance)
        self.assertNotIn("health model flagged this image", guidance)
        self.assertNotIn("Image measurements:", guidance)
        self.assertNotIn("Color screening detected", guidance)
        self.assertNotIn("selected thresholds", guidance)
        self.assertIn("Yellowing detected:", guidance)
        self.assertIn("Slow yellowing on one oldest leaf", guidance)
        self.assertIn("Spreading yellowing or yellowing on new growth", guidance)
        self.assertIn("checking that the potting medium drains", guidance)
        self.assertIn("inspecting roots for soft or mushy areas", guidance)
        self.assertIn("bright, indirect light", guidance)
        self.assertIn("Dark or brown areas detected:", guidance)
        self.assertIn("monitor any dry, unchanged marks", guidance)
        self.assertIn("Soft, wet, or enlarging areas need prompt", guidance)
        self.assertIn("What to do\n1. Not-healthy result:", guidance)
        self.assertNotIn("If a brown/dark area", guidance)
        self.assertNotIn("Track the change:", guidance)
        self.assertNotIn("Cattleya:", guidance)
        self.assertNotIn(
            "Limit: these image measurements and model labels are screening signals",
            guidance,
        )

    def test_visible_symptoms_ignores_green_shadows_and_tiny_color_noise(self):
        image = app.Image.new("RGB", (224, 224), (60, 150, 40))
        pixels = np.asarray(image).copy()
        pixels[60:160, 60:160] = (20, 50, 15)
        pixels[20:24, 20:24] = (220, 190, 30)

        symptoms = app.orchid_ai.analyze_visible_symptoms(app.Image.fromarray(pixels))

        self.assertEqual(symptoms["yellow_percent"], 0.0)
        self.assertEqual(symptoms["brown_dark_percent"], 0.0)

    def test_visible_symptoms_keeps_advice_for_large_yellow_or_brown_patches(self):
        image = app.Image.new("RGB", (224, 224), (60, 150, 40))
        pixels = np.asarray(image).copy()
        pixels[80:140, 80:145] = (220, 190, 30)
        pixels[150:190, 80:145] = (120, 75, 35)

        symptoms = app.orchid_ai.analyze_visible_symptoms(app.Image.fromarray(pixels))

        self.assertGreaterEqual(symptoms["yellow_percent"], 3.0)
        self.assertGreaterEqual(symptoms["brown_dark_percent"], 2.0)

    def test_care_guidance_recommends_checks_for_detected_yellowing_only(self):
        guidance = app.orchid_ai.get_detailed_recommendations(
            "Cattleya",
            "not_healthy",
            98.4,
            93.8,
            {"yellow_percent": 7.0, "brown_dark_percent": 1.2},
        )

        self.assertNotIn("Image measurements:", guidance)
        self.assertIn("checking that the potting medium drains", guidance)
        self.assertIn("bright, indirect light", guidance)
        self.assertNotIn("Dark or brown areas detected:", guidance)

    def test_care_guidance_recommends_direct_care_for_detected_dark_areas(self):
        guidance = app.orchid_ai.get_detailed_recommendations(
            "Phalaenopsis",
            "not_healthy",
            87.0,
            78.0,
            {"yellow_percent": 1.0, "brown_dark_percent": 3.0},
        )

        self.assertIn("Dark or brown areas detected:", guidance)
        self.assertIn("Keep the leaf dry", guidance)
        self.assertIn("Soft, wet, or enlarging areas need prompt", guidance)
        self.assertNotIn("If a brown/dark area", guidance)

    def test_care_guidance_explains_model_flag_when_color_screen_finds_no_marks(self):
        guidance = app.orchid_ai.get_detailed_recommendations(
            "Cattleya",
            "not_healthy",
            89.1,
            99.9,
            {"yellow_percent": 1.8, "brown_dark_percent": 1.5},
        )

        self.assertNotIn("health model flagged this image", guidance)
        self.assertNotIn("color screen found no notable yellow or brown/dark areas", guidance)
        self.assertIn("What to do\n1. Not-healthy result:", guidance)
        self.assertIn("Inspect both sides of the leaves", guidance)
        self.assertIn("If an area is soft, wet, spreading, or enlarging", guidance)

    def test_fallback_guidance_does_not_present_color_guess_as_a_model_prediction(self):
        guidance = app.orchid_ai.get_detailed_recommendations(
            "Dendrobium",
            "healthy",
            75.0,
            75.0,
            {"yellow_percent": 7.0, "brown_dark_percent": 1.1},
            using_real_model=False,
        )

        self.assertIn("no trained model is available", guidance)
        self.assertIn("rough color-based estimates", guidance)
        self.assertNotIn("Color screening detected", guidance)
        self.assertIn("cannot identify the cause", guidance)
        self.assertNotIn("fallback estimated Dendrobium", guidance)
        self.assertNotIn("75.0% confidence", guidance)

    def test_fallback_prediction_returns_normalized_model_input_preview(self):
        image_buffer = io.BytesIO()
        app.Image.new("RGB", (32, 24), color=(30, 140, 60)).save(image_buffer, format="PNG")
        image_data = "data:image/png;base64," + base64.b64encode(
            image_buffer.getvalue()
        ).decode("ascii")

        original_remove_background = app.orchid_ai.remove_background
        original_analyze_symptoms = app.orchid_ai.analyze_visible_symptoms
        original_augmentation_previews = app.orchid_ai.create_training_augmentation_previews
        app.orchid_ai.remove_background = lambda image: image
        app.orchid_ai.analyze_visible_symptoms = lambda image: {
            "yellow_percent": 0.0,
            "brown_dark_percent": 0.0,
        }
        app.orchid_ai.create_training_augmentation_previews = lambda image: [
            {
                "name": "Rotation",
                "configuration": "RandomRotation(0.1)",
                "preview_image": "data:image/png;base64,dGVzdA==",
            }
        ]
        try:
            result = app.orchid_ai.fallback_analysis(image_data)
        finally:
            app.orchid_ai.remove_background = original_remove_background
            app.orchid_ai.analyze_visible_symptoms = original_analyze_symptoms
            app.orchid_ai.create_training_augmentation_previews = original_augmentation_previews

        self.assertTrue(result["success"])
        self.assertTrue(result["model_input_image"].startswith("data:image/png;base64,"))
        self.assertAlmostEqual(result["normalized_range"][0], 30 / 255)
        self.assertAlmostEqual(result["normalized_range"][1], 140 / 255)
        self.assertEqual(result["rgb_channel_averages"], {"R": 30.0, "G": 140.0, "B": 60.0})
        self.assertEqual(result["visible_symptoms"]["yellow_percent"], 0.0)
        self.assertEqual(result["training_augmentation_previews"][0]["name"], "Rotation")
        self.assertIsNone(result["training_augmentation_preview_error"])
        self.assertIn("Color-based fallback estimate", result["classifier_pipeline"])
        self.assertIsNone(result["confidence"])

    def test_model_metrics_endpoint_uses_local_file_without_mysql_password(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = os.path.join(temp_dir, "models")
            os.makedirs(models_dir)
            metrics_file = os.path.join(models_dir, "training_metrics.json")
            expected_metrics = {"accuracy": 0.84, "status": "completed"}
            with open(metrics_file, "w", encoding="utf-8") as handle:
                json.dump(expected_metrics, handle)

            original_base_dir = app.BASE_DIR
            original_mysql_password = app.MYSQL_CONFIG["password"]
            app.BASE_DIR = temp_dir
            app.MYSQL_CONFIG["password"] = ""
            try:
                client = app.app.test_client()
                with client.session_transaction() as session:
                    session["logged_in"] = True
                response = client.get("/api/admin/model-metrics")
            finally:
                app.BASE_DIR = original_base_dir
                app.MYSQL_CONFIG["password"] = original_mysql_password

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["metrics"]["accuracy"], expected_metrics["accuracy"])
        self.assertFalse(payload["metrics"]["evaluation_is_current"])

    def test_model_metrics_endpoint_reports_stale_training_as_interrupted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = os.path.join(temp_dir, "models")
            os.makedirs(models_dir)
            metrics_file = os.path.join(models_dir, "training_metrics.json")
            expected_metrics = {"accuracy": "--", "status": "training"}
            with open(metrics_file, "w", encoding="utf-8") as handle:
                json.dump(expected_metrics, handle)

            original_base_dir = app.BASE_DIR
            original_mysql_password = app.MYSQL_CONFIG["password"]
            original_progress = dict(app.training_progress)
            app.BASE_DIR = temp_dir
            app.MYSQL_CONFIG["password"] = ""
            app.training_progress["status"] = "idle"
            try:
                client = app.app.test_client()
                with client.session_transaction() as session:
                    session["logged_in"] = True
                response = client.get("/api/admin/model-metrics")

                app.training_progress["status"] = "training"
                active_response = client.get("/api/admin/model-metrics")
            finally:
                app.BASE_DIR = original_base_dir
                app.MYSQL_CONFIG["password"] = original_mysql_password
                app.training_progress.clear()
                app.training_progress.update(original_progress)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["metrics"]["status"], "interrupted")
        self.assertEqual(active_response.status_code, 200)
        self.assertEqual(active_response.get_json()["metrics"]["status"], "training")

    def test_demo_login_uses_local_users_when_mysql_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            users_file = os.path.join(temp_dir, "users.json")
            with open(users_file, "w", encoding="utf-8") as handle:
                json.dump({"demo": {"password": "demo123"}}, handle)

            original_base_dir = app.BASE_DIR
            original_mysql_password = app.MYSQL_CONFIG["password"]
            original_get_database_connection = app.get_database_connection

            def unavailable_database():
                raise app.mysql.connector.Error("Database unavailable")

            app.BASE_DIR = temp_dir
            app.MYSQL_CONFIG["password"] = "configured"
            app.get_database_connection = unavailable_database
            try:
                response = app.app.test_client().post(
                    "/login",
                    data={"username": "demo", "password": "demo123"},
                )
            finally:
                app.BASE_DIR = original_base_dir
                app.MYSQL_CONFIG["password"] = original_mysql_password
                app.get_database_connection = original_get_database_connection

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, "/user/dashboard")

    def test_create_account_saves_locally_when_mysql_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_base_dir = app.BASE_DIR
            original_mysql_password = app.MYSQL_CONFIG["password"]
            original_get_database_connection = app.get_database_connection

            def unavailable_database():
                raise app.mysql.connector.Error("Database unavailable")

            app.BASE_DIR = temp_dir
            app.MYSQL_CONFIG["password"] = "configured"
            app.get_database_connection = unavailable_database
            try:
                response = app.app.test_client().post(
                    "/create-account",
                    data={
                        "username": "newuser",
                        "password": "secure123",
                        "confirm_password": "secure123",
                    },
                )
                with open(os.path.join(temp_dir, "users.json"), encoding="utf-8") as handle:
                    users = json.load(handle)
            finally:
                app.BASE_DIR = original_base_dir
                app.MYSQL_CONFIG["password"] = original_mysql_password
                app.get_database_connection = original_get_database_connection

        self.assertEqual(response.status_code, 200)
        self.assertEqual(users["newuser"]["password"], "secure123")


if __name__ == "__main__":
    unittest.main()
