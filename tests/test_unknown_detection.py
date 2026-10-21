import unittest

import numpy as np

from unknown_detection import ValidationCalibratedDetector


class ValidationCalibratedDetectorTest(unittest.TestCase):
    def make_detector(self):
        classes = ["Cattleya", "Vanda"]
        train_features = np.array(
            [[0.0, 0.0], [0.1, -0.1], [-0.1, 0.1], [10.0, 10.0], [10.1, 9.9], [9.9, 10.1]]
        )
        train_labels = np.array([0, 0, 0, 1, 1, 1])
        base_validation_features = np.array(
            [[0.05, 0.0], [-0.05, 0.05], [0.0, -0.05], [0.08, 0.04],
             [10.05, 10.0], [9.95, 10.05], [10.0, 9.95], [9.92, 9.98]]
        )
        base_species_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        base_health_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        base_species_probabilities = np.array(
            [[0.95, 0.05]] * 4 + [[0.05, 0.95]] * 4
        )
        base_health_probabilities = np.array(
            [[0.95, 0.05], [0.05, 0.95]] * 4
        )
        validation_features = np.tile(base_validation_features, (5, 1))
        species_labels = np.tile(base_species_labels, 5)
        health_labels = np.tile(base_health_labels, 6)
        species_probabilities = np.tile(base_species_probabilities, (5, 1))
        health_probabilities = np.tile(base_health_probabilities, (6, 1))
        detector = ValidationCalibratedDetector(alpha=0.1)
        detector.fit(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels={
                "species": species_labels,
                "health": health_labels,
            },
            validation_probabilities={
                "species": species_probabilities,
                "health": health_probabilities,
            },
            probability_classes={
                "species": classes,
                "health": ["healthy", "not_healthy"],
            },
            validation_sharpness=np.tile(
                np.array([12.0, 14.0, 15.0, 13.0, 10.0, 11.0, 13.0, 12.0]),
                5,
            ),
            classes=classes,
        )
        return detector

    def test_accepts_validation_like_feature_and_rejects_far_outlier(self):
        detector = self.make_detector()

        self.assertEqual(
            detector.is_out_of_distribution(np.array([0.02, 0.03]), "Cattleya")[0],
            False,
        )
        self.assertTrue(
            detector.is_out_of_distribution(np.array([100.0, -100.0]), "Cattleya")[0]
        )

    def test_prediction_sets_use_validation_scores_for_each_task(self):
        detector = self.make_detector()

        self.assertEqual(
            detector.prediction_set(
                detector.calibrate_probabilities(np.array([[0.99, 0.01]]), "species")[0],
                "species",
            ),
            ["Cattleya"],
        )
        self.assertEqual(
            detector.prediction_set(
                detector.calibrate_probabilities(np.array([[0.99, 0.01]]), "health")[0],
                "health",
            ),
            ["healthy"],
        )

    def test_temperature_calibration_reduces_overconfident_validation_scores(self):
        detector = self.make_detector()
        probabilities = np.array([[0.999, 0.001], [0.999, 0.001]])
        detector.probability_temperatures["health"] = detector._fit_temperature(
            probabilities,
            np.array([1, 1]),
        )

        calibrated = detector.calibrate_probabilities(probabilities, "health")

        self.assertTrue(detector.has_probability_calibration("health"))
        self.assertTrue(np.all(calibrated[:, 0] < probabilities[:, 0]))
        self.assertTrue(np.allclose(calibrated.sum(axis=1), 1.0))

    def test_prediction_sets_pool_small_class_calibration_samples(self):
        detector = ValidationCalibratedDetector(alpha=0.1)
        detector.probability_scores["species"] = {
            "Cattleya": [0.01] * 5,
            "Vanda": [0.01] * 5,
        }

        self.assertEqual(
            detector.prediction_set(
                detector.calibrate_probabilities(np.array([[0.01, 0.995]]), "species")[0],
                "species",
            ),
            ["Vanda"],
        )

    def test_blur_cutoff_is_calibrated_from_validation_images(self):
        detector = self.make_detector()

        self.assertFalse(detector.is_too_blurry(10.0))
        self.assertTrue(detector.is_too_blurry(9.99))

    def test_detector_round_trip_preserves_predictions(self):
        detector = self.make_detector()
        loaded = ValidationCalibratedDetector.from_dict(detector.to_dict())

        self.assertEqual(
            loaded.prediction_set(
                loaded.calibrate_probabilities(
                    np.array([[0.99, 0.01]]),
                    "species",
                )[0],
                "species",
            ),
            ["Cattleya"],
        )
        self.assertEqual(
            loaded.probability_temperatures,
            detector.probability_temperatures,
        )
        np.testing.assert_allclose(
            loaded.calibrate_probabilities(np.array([[0.8, 0.2]]), "health"),
            detector.calibrate_probabilities(np.array([[0.8, 0.2]]), "health"),
        )
        self.assertEqual(
            loaded.is_out_of_distribution(np.array([100.0, -100.0]), "Cattleya"),
            detector.is_out_of_distribution(np.array([100.0, -100.0]), "Cattleya"),
        )


if __name__ == "__main__":
    unittest.main()
