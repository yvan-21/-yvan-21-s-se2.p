import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from tools.validate_dataset_with_plantnet import (
    API_URL,
    ValidationError,
    build_report_row,
    call_plantnet,
    get_labeled_images,
    main,
    parse_args,
    parse_plantnet_response,
    prepare_image,
)


class PlantNetDatasetValidationTest(unittest.TestCase):
    def test_image_preparation_resizes_and_converts_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "orchid.png"
            Image.new("RGBA", (2400, 1200), (20, 130, 80, 255)).save(source)
            original_bytes = source.read_bytes()

            prepared = prepare_image(source)
            with Image.open(io.BytesIO(prepared)) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (1280, 640))
            self.assertEqual(source.read_bytes(), original_bytes)

    def test_dataset_discovery_uses_folder_species_and_normalized_health(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Cattleya" / "not_healthy" / "leaf.jpg"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (20, 20)).save(path)

            images = list(get_labeled_images(temp_dir))

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0][1:], ("Cattleya", "not healthy"))

    def test_report_flags_species_disagreements_without_mutating_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "datasets"
            image_path = dataset_dir / "Cattleya" / "healthy" / "leaf.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (20, 20)).save(image_path)

            row = build_report_row(
                image_path,
                dataset_dir,
                "Cattleya",
                {
                    "species_suggestion": "Vanda",
                    "scientific_name": "Vanda tricolor",
                    "score": 0.81,
                },
            )

        self.assertEqual(row["dataset_species_label"], "Cattleya")
        self.assertEqual(row["species_agrees"], "false")
        self.assertEqual(row["review_recommended"], "true")
        self.assertIn("differs from folder label", row["review_reason"])
        self.assertNotIn("health", row)

    def test_plantnet_request_sends_key_and_image_as_form_data(self):
        response = Mock(
            ok=True,
            json=lambda: {
                "results": [{
                    "score": 0.8,
                    "species": {"scientificNameWithoutAuthor": "Cattleya labiata"},
                }]
            },
        )
        session = Mock()
        session.post.return_value = response

        actual = call_plantnet(b"jpeg-bytes", "test-secret", session=session)

        self.assertEqual(actual["species_suggestion"], "Cattleya")
        self.assertEqual(actual["scientific_name"], "Cattleya labiata")
        session.post.assert_called_once()
        args, kwargs = session.post.call_args
        self.assertEqual(args[0], API_URL)
        self.assertEqual(kwargs["params"], {"api-key": "test-secret"})
        self.assertEqual(
            kwargs["files"]["images"],
            ("orchid.jpg", b"jpeg-bytes", "image/jpeg"),
        )
        self.assertEqual(kwargs["data"], {"organs": "leaf"})
        self.assertEqual(kwargs["timeout"], 90)

    def test_http_errors_do_not_include_response_body_or_key(self):
        session = Mock()
        session.post.return_value = Mock(ok=False, status_code=403, text="secret body")

        with self.assertRaises(ValidationError) as context:
            call_plantnet(b"jpeg-bytes", "test-secret", session=session)

        self.assertIn("HTTP 403", str(context.exception))
        self.assertNotIn("secret body", str(context.exception))
        self.assertNotIn("test-secret", str(context.exception))

    def test_parses_top_result_genus_and_other_genera(self):
        response = {
            "results": [{
                "score": 0.75,
                "species": {"scientificNameWithoutAuthor": "Cattleya labiata"},
            }]
        }
        self.assertEqual(
            parse_plantnet_response(response),
            {
                "species_suggestion": "Cattleya",
                "scientific_name": "Cattleya labiata",
                "score": 0.75,
            },
        )
        response["results"][0]["species"]["scientificNameWithoutAuthor"] = "Acer rubrum"
        self.assertEqual(
            parse_plantnet_response(response)["species_suggestion"],
            "other_genus",
        )

    def test_empty_or_invalid_results_are_handled_explicitly(self):
        self.assertEqual(
            parse_plantnet_response({"results": []})["species_suggestion"],
            "uncertain",
        )
        with self.assertRaises(ValueError):
            parse_plantnet_response({
                "results": [{
                    "score": float("nan"),
                    "species": {"scientificNameWithoutAuthor": "Vanda tricolor"},
                }]
            })

    def test_upload_requires_explicit_confirmation_and_configured_key(self):
        original_key = os.environ.pop("PLANTNET_API_KEY", None)
        try:
            exit_code = main([
                "--dataset",
                "missing",
                "--output",
                "report.csv",
                "--confirm-external-upload",
            ])
        finally:
            if original_key is not None:
                os.environ["PLANTNET_API_KEY"] = original_key

        self.assertEqual(exit_code, 1)

    def test_upload_requires_explicit_confirmation_flag(self):
        with self.assertRaises(SystemExit) as context:
            parse_args(["--dataset", "datasets"])

        self.assertEqual(context.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
