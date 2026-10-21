import os
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

sys.modules["tensorflow"].get_logger = lambda: types.SimpleNamespace(
    setLevel=lambda *args, **kwargs: None
)
sys.modules["tensorflow"].keras = types.SimpleNamespace(
    callbacks=types.SimpleNamespace(Callback=type("DummyCallback", (), {}))
)
sys.modules["keras"].layers = types.SimpleNamespace()
sys.modules["keras"].models = types.SimpleNamespace()
sys.modules["sklearn.model_selection"].StratifiedGroupKFold = type(
    "Dummy", (), {}
)
sys.modules["sklearn.metrics"].accuracy_score = lambda *args, **kwargs: 0.0
sys.modules["sklearn.metrics"].classification_report = lambda *args, **kwargs: ""
sys.modules["sklearn.metrics"].confusion_matrix = lambda *args, **kwargs: None
sys.modules["sklearn.metrics"].log_loss = lambda *args, **kwargs: 0.0
sys.modules["sklearn.metrics"].precision_recall_fscore_support = (
    lambda *args, **kwargs: ([], [], [], [])
)
sys.modules["sklearn.linear_model"].LogisticRegression = type("Dummy", (), {})
sys.modules["sklearn.pipeline"].make_pipeline = lambda *args, **kwargs: None
sys.modules["sklearn.preprocessing"].StandardScaler = type("Dummy", (), {})
sys.modules["joblib"].dump = lambda *args, **kwargs: None
sys.modules["joblib"].load = lambda *args, **kwargs: None
sys.modules["skimage.feature"].graycomatrix = lambda *args, **kwargs: None
sys.modules["skimage.feature"].graycoprops = lambda *args, **kwargs: None

import app


class AdminLoginConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()
        self.username_patch = patch.object(app, "ADMIN_USERNAME", "admin")
        self.password_patch = patch.object(
            app, "ADMIN_PASSWORD", "test-admin-password"
        )
        self.username_patch.start()
        self.password_patch.start()
        self.addCleanup(self.password_patch.stop)
        self.addCleanup(self.username_patch.stop)

    def test_admin_can_log_in_with_configured_credentials(self):
        response = self.client.post(
            "/login",
            data={"username": "admin", "password": "test-admin-password"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/dashboard", response.headers["Location"])

    def test_unconfigured_admin_login_shows_setup_instructions(self):
        with patch.object(app, "ADMIN_PASSWORD", None):
            response = self.client.get("/login")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Admin login is not configured", response.data)
        self.assertIn(b"Set ADMIN_PASSWORD in .env", response.data)

    def test_admin_login_reports_missing_password_configuration(self):
        with patch.object(app, "ADMIN_PASSWORD", None):
            response = self.client.post(
                "/login",
                data={"username": "admin", "password": "test-admin-password"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Admin login is not configured", response.data)

    def test_incorrect_admin_password_does_not_authenticate(self):
        with patch.object(app, "load_app_document", return_value={}):
            response = self.client.post(
                "/login",
                data={"username": "admin", "password": "incorrect"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Invalid username or password", response.data)


if __name__ == "__main__":
    unittest.main()
