import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import preprocess_dataset


class PreprocessDatasetTest(unittest.TestCase):
    def test_centered_square_view_crops_original_before_resizing(self):
        image = Image.new("RGB", (320, 180))
        for x in range(320):
            color = (240, 20, 20) if x < 70 or x >= 250 else (20, 20, 240)
            for y in range(180):
                image.putpixel((x, y), color)

        square_view = preprocess_dataset._centered_square_species_view(image)

        self.assertEqual(square_view.size, (224, 224))
        self.assertGreater(square_view.getpixel((10, 112))[2], square_view.getpixel((10, 112))[0])
        self.assertGreater(square_view.getpixel((213, 112))[2], square_view.getpixel((213, 112))[0])

    def test_zoom_out_extends_edge_pixels_instead_of_adding_black_borders(self):
        image = Image.new("RGB", (20, 20), color=(90, 140, 70))

        zoomed = preprocess_dataset._zoom_image(image, 0.5)

        self.assertEqual(zoomed.size, (20, 20))
        self.assertEqual(zoomed.getpixel((0, 0)), (90, 140, 70))
        self.assertEqual(zoomed.getpixel((19, 19)), (90, 140, 70))

    def test_every_augmented_output_is_horizontally_flipped(self):
        class IdentityAugmenter:
            def flow(self, images, batch_size, shuffle):
                assert batch_size == 1
                assert not shuffle
                for _ in range(3):
                    yield images

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "datasets"
            cleaned_dir = root / "cleaned_dataset"
            augmented_dir = root / "augmented_dataset"
            stages_dir = root / "preprocessing_stages"
            source_path = dataset_dir / "Dendrobium" / "healthy" / "asymmetric.png"
            source_path.parent.mkdir(parents=True, exist_ok=True)

            image = Image.new("RGB", (320, 180))
            for x in range(320):
                color = (240, 20, 20) if x < 160 else (20, 20, 240)
                for y in range(180):
                    image.putpixel((x, y), color)
            image.save(source_path)

            with (
                patch.object(preprocess_dataset, "DATASET_DIR", dataset_dir),
                patch.object(preprocess_dataset, "CLEANED_DIR", cleaned_dir),
                patch.object(preprocess_dataset, "AUGMENTED_DIR", augmented_dir),
                patch.object(preprocess_dataset, "STAGES_DIR", stages_dir),
                patch.object(preprocess_dataset, "augmenter", IdentityAugmenter()),
                patch.object(
                    preprocess_dataset,
                    "remove_background",
                    side_effect=lambda image: image.copy(),
                ),
            ):
                result = preprocess_dataset.preprocess_dataset()

            self.assertEqual(result["augmented"], 3)
            for augmentation_index in range(1, 4):
                output_path = (
                    augmented_dir
                    / "Dendrobium"
                    / "healthy"
                    / f"asymmetric_aug_{augmentation_index:02d}.jpg"
                )
                with Image.open(output_path) as output:
                    left_pixel = output.convert("RGB").getpixel((30, 112))
                    right_pixel = output.convert("RGB").getpixel((190, 112))
                self.assertGreater(left_pixel[2], left_pixel[0])
                self.assertGreater(right_pixel[0], right_pixel[2])

    def test_outputs_are_jpeg_resized_and_reruns_do_not_duplicate_augmentations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "datasets"
            cleaned_dir = root / "cleaned_dataset"
            augmented_dir = root / "augmented_dataset"
            stages_dir = root / "preprocessing_stages"
            source_paths = [
                dataset_dir / "Cattleya" / "healthy" / "leaf.jpg",
                dataset_dir / "Vanda" / "not healthy" / "leaf.webp",
            ]

            for source_path in source_paths:
                source_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (320, 180), color=(90, 140, 70)).save(source_path)

            original_bytes = [source_path.read_bytes() for source_path in source_paths]

            stale_cleaned_path = cleaned_dir / "Vanda" / "not healthy" / "leaf.webp"
            stale_cleaned_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (224, 224)).save(stale_cleaned_path, format="JPEG")
            stale_augmented_path = (
                augmented_dir / "Vanda" / "not healthy" / "leaf_aug_0_1234.jpg"
            )
            stale_augmented_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (224, 224)).save(stale_augmented_path)

            with (
                patch.object(preprocess_dataset, "DATASET_DIR", dataset_dir),
                patch.object(preprocess_dataset, "CLEANED_DIR", cleaned_dir),
                patch.object(preprocess_dataset, "AUGMENTED_DIR", augmented_dir),
                patch.object(preprocess_dataset, "STAGES_DIR", stages_dir),
                patch.object(
                    preprocess_dataset,
                    "remove_background",
                    side_effect=lambda image: image.copy(),
                ),
            ):
                first_run = preprocess_dataset.preprocess_dataset()
                second_run = preprocess_dataset.preprocess_dataset()

            cleaned_images = sorted(cleaned_dir.rglob("*.jpg"))
            augmented_images = sorted(augmented_dir.rglob("*.jpg"))
            self.assertEqual(first_run, {
                "originals": 2,
                "processed": 2,
                "augmented": 6,
                "errors": 0,
                "background_errors": 0,
            })
            self.assertEqual(second_run, first_run)
            self.assertEqual(len(cleaned_images), 2)
            self.assertEqual(len(augmented_images), 6)
            stage_names = (
                "resized",
                "centered_square_species_view",
                "random_rotation",
                "flipping",
                "zooming",
                "brightness_adjustment",
                "background_removal",
            )
            for stage_name in stage_names:
                stage_images = sorted((stages_dir / stage_name).rglob("*.jpg"))
                self.assertEqual(len(stage_images), 2, stage_name)
            self.assertFalse(stale_cleaned_path.exists())
            self.assertFalse(stale_augmented_path.exists())
            self.assertEqual(
                [source_path.read_bytes() for source_path in source_paths],
                original_bytes,
            )

            for image_path in cleaned_images + augmented_images:
                with Image.open(image_path) as image:
                    self.assertEqual(image.format, "JPEG")
                    self.assertEqual(image.size, (224, 224))
                    self.assertEqual(image.mode, "RGB")

            self.assertTrue(
                (cleaned_dir / "Vanda" / "not healthy" / "leaf.jpg").is_file()
            )


if __name__ == "__main__":
    unittest.main()
