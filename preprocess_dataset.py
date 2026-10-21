"""Create resized JPEGs and augmented previews without modifying source images.

The augmented output contains derivatives of every source image. Do not use it
as a train/validation/test dataset: split original images by plant/session first
and augment only the training partition. The training pipeline should normalize
pixel values when loading images rather than saving normalized JPEGs.

The preprocessing_stages folder contains separate visual examples for each
operation. These examples are for inspection and must not be used as extra
training samples.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array  # pyright: ignore[reportMissingImports]
from background_removal import remove_background


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
CLEANED_DIR = BASE_DIR / "cleaned_dataset"
AUGMENTED_DIR = BASE_DIR / "augmented_dataset"
STAGES_DIR = BASE_DIR / "preprocessing_stages"

IMAGE_SIZE = (224, 224)
AUGMENTATIONS_PER_IMAGE = 3
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

augmenter = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=False,
    vertical_flip=False,
    zoom_range=0.15,
    brightness_range=(0.8, 1.2),
    fill_mode="nearest",
)


def _remove_previous_augmentations(folder: Path, source_stem: str) -> None:
    """Remove only augmentation files previously generated for this source."""
    prefix = f"{source_stem}_aug_"
    if folder.exists():
        for path in folder.iterdir():
            if path.is_file() and path.name.startswith(prefix) and path.suffix.lower() == ".jpg":
                path.unlink()


def _save_stage(image: Image.Image, stage_name: str, relative_path: Path) -> None:
    output_path = STAGES_DIR / stage_name / relative_path.with_suffix(".jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=95)


def _zoom_image(image: Image.Image, factor: float) -> Image.Image:
    width, height = image.size
    resized_size = (round(width * factor), round(height * factor))
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    if factor >= 1:
        left = (resized.width - width) // 2
        top = (resized.height - height) // 2
        return resized.crop((left, top, left + width, top + height))

    left = (width - resized.width) // 2
    top = (height - resized.height) // 2
    pixels = np.asarray(resized)
    padded_pixels = np.pad(
        pixels,
        (
            (top, height - resized.height - top),
            (left, width - resized.width - left),
            (0, 0),
        ),
        mode="edge",
    )
    return Image.fromarray(padded_pixels)


def _centered_square_species_view(image: Image.Image) -> Image.Image:
    """Match the prediction pipeline's centered square crop and model resize."""
    width, height = image.size
    crop_side = min(width, height)
    left = (width - crop_side) // 2
    top = (height - crop_side) // 2
    return image.crop((left, top, left + crop_side, top + crop_side)).resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS,
    )


def preprocess_dataset() -> dict[str, int]:
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    image_paths = sorted(
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS
    )
    if not image_paths:
        raise ValueError(f"No supported images found in dataset folder: {DATASET_DIR}")

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_augmented = 0
    total_errors = 0
    total_background_errors = 0

    print(f"Dataset folder: {DATASET_DIR}")
    print(f"Images found: {len(image_paths)}")
    print("Starting image preprocessing...\n")

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(DATASET_DIR)
        cleaned_path = (CLEANED_DIR / relative_path).with_suffix(".jpg")
        augmented_folder = AUGMENTED_DIR / relative_path.parent

        try:
            with Image.open(image_path) as original:
                source_image = ImageOps.exif_transpose(original).convert("RGB")
                square_species_view = _centered_square_species_view(source_image)
                image = source_image
                image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

            cleaned_path.parent.mkdir(parents=True, exist_ok=True)
            augmented_folder.mkdir(parents=True, exist_ok=True)
            image.save(cleaned_path, format="JPEG", quality=95)
            _save_stage(image, "resized", relative_path)
            _save_stage(
                square_species_view,
                "centered_square_species_view",
                relative_path,
            )
            _save_stage(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), "flipping", relative_path)

            rotated = image.rotate(
                random.uniform(-20, 20),
                resample=Image.Resampling.BICUBIC,
                fillcolor=(0, 0, 0),
            )
            _save_stage(rotated, "random_rotation", relative_path)
            _save_stage(_zoom_image(image, random.uniform(0.85, 1.15)), "zooming", relative_path)
            brightness_factor = random.uniform(0.8, 1.2)
            _save_stage(
                ImageEnhance.Brightness(image).enhance(brightness_factor),
                "brightness_adjustment",
                relative_path,
            )
            try:
                _save_stage(remove_background(image), "background_removal", relative_path)
            except Exception as error:
                total_background_errors += 1
                print(f"[BACKGROUND REMOVAL ERROR] {image_path}: {error}")

            # Remove the old incorrectly suffixed output from earlier versions.
            previous_cleaned_path = CLEANED_DIR / relative_path
            if previous_cleaned_path != cleaned_path and previous_cleaned_path.exists():
                previous_cleaned_path.unlink()

            image_array = img_to_array(image)
            image_batch = np.expand_dims(image_array, axis=0)
            generated = augmenter.flow(
                image_batch,
                batch_size=1,
                shuffle=False,
            )

            _remove_previous_augmentations(augmented_folder, image_path.stem)
            for augmentation_index in range(1, AUGMENTATIONS_PER_IMAGE + 1):
                augmented_array = next(generated)[0]
                augmented_image = Image.fromarray(
                    np.clip(augmented_array, 0, 255).astype(np.uint8)
                ).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                augmented_image.save(
                    augmented_folder / f"{image_path.stem}_aug_{augmentation_index:02d}.jpg",
                    format="JPEG",
                    quality=95,
                )
                total_augmented += 1

            total_processed += 1
            print(f"[{index}/{len(image_paths)}] Processed: {relative_path}")
        except (OSError, ValueError, UnidentifiedImageError) as error:
            total_errors += 1
            print(f"[ERROR] {image_path}: {error}")

    print("\n================================")
    print("IMAGE PREPROCESSING COMPLETE")
    print("================================")
    print(f"Original images found: {len(image_paths)}")
    print(f"Successfully processed: {total_processed}")
    print(f"Augmented images created: {total_augmented}")
    print(f"Errors: {total_errors}")
    print(f"Background-removal errors: {total_background_errors}")
    print(f"\nCleaned images: {CLEANED_DIR}")
    print(f"Augmented images: {AUGMENTED_DIR}")
    print(f"Step-by-step stage images: {STAGES_DIR}")

    return {
        "originals": len(image_paths),
        "processed": total_processed,
        "augmented": total_augmented,
        "errors": total_errors,
        "background_errors": total_background_errors,
    }


if __name__ == "__main__":
    preprocess_dataset()
