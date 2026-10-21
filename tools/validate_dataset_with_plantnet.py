"""Create a non-destructive Pl@ntNet species cross-check report."""

import argparse
import csv
import io
import math
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps


SPECIES_LABELS = ("Cattleya", "Dendrobium", "Phalaenopsis", "Vanda")
HEALTH_LABELS = ("healthy", "not healthy", "not_healthy")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
API_URL = "https://my-api.plantnet.org/v2/identify/all"
OUTPUT_FIELDS = (
    "image_path",
    "dataset_species_label",
    "plantnet_species_suggestion",
    "plantnet_top_scientific_name",
    "plantnet_top_match_score_uncalibrated",
    "species_agrees",
    "review_recommended",
    "review_reason",
    "provider",
)


class ValidationError(RuntimeError):
    """Raised when an image or Pl@ntNet response cannot be validated safely."""


def prepare_image(image_path):
    """Apply EXIF orientation and resize an image for review without changing source."""
    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=85, optimize=True)
    except (OSError, ValueError) as error:
        raise ValidationError(f"Could not read image {image_path}: {error}") from error
    return output.getvalue()


def call_plantnet(image_bytes, api_key, session=None, timeout=90):
    """Request a Pl@ntNet identification result without logging the API key."""
    client = session or requests
    try:
        response = client.post(
            API_URL,
            params={"api-key": api_key},
            files={"images": ("orchid.jpg", image_bytes, "image/jpeg")},
            data={"organs": "leaf"},
            timeout=timeout,
        )
    except requests.RequestException as error:
        raise ValidationError(
            f"Pl@ntNet request failed ({type(error).__name__}). Check network "
            "access and retry; the credential is not included in this message."
        ) from None

    if not response.ok:
        raise ValidationError(
            f"Pl@ntNet returned HTTP {response.status_code}. Check API key access "
            "and quota; response body omitted."
        )

    try:
        return parse_plantnet_response(response.json())
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise ValidationError(
            "Pl@ntNet returned a response that did not match the expected format."
        ) from None


def parse_plantnet_response(response_data):
    results = response_data["results"]
    if not isinstance(results, list):
        raise ValueError("Invalid results list")
    if not results:
        return {
            "species_suggestion": "uncertain",
            "scientific_name": "",
            "score": 0.0,
        }

    top_result = results[0]
    species = top_result["species"]
    scientific_name = species["scientificNameWithoutAuthor"]
    if not isinstance(scientific_name, str) or not scientific_name.strip():
        raise ValueError("Invalid scientific name")
    score = top_result["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("Invalid match score")
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Match score out of range")

    genus = scientific_name.strip().split()[0]
    suggestion = genus if genus in SPECIES_LABELS else "other_genus"
    return {
        "species_suggestion": suggestion,
        "scientific_name": scientific_name.strip(),
        "score": score,
    }


def get_labeled_images(dataset_dir):
    """Yield supported image paths and labels in the app's species/health folders."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise ValidationError(f"Dataset directory does not exist: {dataset_dir}")
    for image_path in sorted(dataset_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            species, health = image_path.relative_to(dataset_dir).parts[:2]
        except ValueError as error:
            raise ValidationError(f"Image is outside dataset root: {image_path}") from error
        if species not in SPECIES_LABELS or health not in HEALTH_LABELS:
            raise ValidationError(
                f"Unsupported dataset label path: {image_path.relative_to(dataset_dir)}"
            )
        yield image_path, species, "healthy" if health == "healthy" else "not healthy"


def build_report_row(image_path, dataset_dir, species_label, result):
    species_suggestion = result["species_suggestion"]
    species_agrees = species_suggestion == species_label
    low_score = result["score"] < 0.65
    review_recommended = (
        not species_agrees
        or low_score
        or species_suggestion == "uncertain"
    )
    reasons = []
    if not species_agrees:
        reasons.append("species suggestion differs from folder label")
    if low_score:
        reasons.append("low uncalibrated top-match score")
    if species_suggestion in {"uncertain", "other_genus"}:
        reasons.append("top match is uncertain or outside the dataset genera")

    return {
        "image_path": image_path.relative_to(dataset_dir).as_posix(),
        "dataset_species_label": species_label,
        "plantnet_species_suggestion": species_suggestion,
        "plantnet_top_scientific_name": result["scientific_name"],
        "plantnet_top_match_score_uncalibrated": f"{result['score']:.3f}",
        "species_agrees": str(species_agrees).lower(),
        "review_recommended": str(review_recommended).lower(),
        "review_reason": "; ".join(reasons),
        "provider": "Pl@ntNet",
    }


def write_report_row(writer, row, report_file):
    writer.writerow(row)
    report_file.flush()


def run_validation(args):
    api_key = os.environ.get("PLANTNET_API_KEY", "").strip()
    if not api_key:
        raise ValidationError(
            "PLANTNET_API_KEY is not set. Set it locally in the shell; do not put it "
            "in source files or command-line arguments."
        )

    dataset_dir = Path(args.dataset).resolve()
    images = list(get_labeled_images(dataset_dir))
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise ValidationError(f"No supported labeled images found in {dataset_dir}")

    output_path = Path(args.output).resolve()
    if output_path == dataset_dir or dataset_dir in output_path.parents:
        raise ValidationError("Choose a report output path outside the dataset directory.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.resume and output_path.exists() else "w"
    completed_paths = set()
    if mode == "a":
        with output_path.open("r", newline="", encoding="utf-8-sig") as existing:
            completed_paths = {
                row["image_path"]
                for row in csv.DictReader(existing)
                if row.get("image_path")
            }

    write_header = mode == "w" or output_path.stat().st_size == 0
    completed_count = len(completed_paths)
    with output_path.open(mode, newline="", encoding="utf-8-sig") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()
            report_file.flush()

        for index, (image_path, species_label, _health_label) in enumerate(images, start=1):
            relative_path = image_path.relative_to(dataset_dir).as_posix()
            if relative_path in completed_paths:
                continue
            print(f"Identifying {index}/{len(images)}: {relative_path}", flush=True)
            image_bytes = prepare_image(image_path)
            result = call_plantnet(image_bytes, api_key)
            row = build_report_row(
                image_path,
                dataset_dir,
                species_label,
                result,
            )
            write_report_row(writer, row, report_file)
            completed_count += 1
            if index < len(images):
                time.sleep(args.delay)

    print(
        f"Wrote {completed_count} review row(s) to {output_path}. "
        "Review flags are suggestions only; no dataset labels were changed."
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Send resized copies of dataset images to Pl@ntNet for a supplementary "
            "species-genus cross-check. This does not assess plant health."
        )
    )
    parser.add_argument("--dataset", default="datasets", help="Dataset folder.")
    parser.add_argument(
        "--output",
        default="dataset_validation_plantnet.csv",
        help="Review report CSV path (kept outside the dataset folder).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum images to send; default 5. Use 0 to review all images.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between API calls (default 1).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip image paths already present in the output CSV.",
    )
    parser.add_argument(
        "--confirm-external-upload",
        action="store_true",
        help="Confirm resized dataset images may be sent to Pl@ntNet.",
    )
    args = parser.parse_args(argv)
    if not args.confirm_external_upload:
        parser.error(
            "Image upload is opt-in. Add --confirm-external-upload to send images to Pl@ntNet."
        )
    if args.limit < 0 or args.delay < 0:
        parser.error("--limit and --delay must not be negative.")
    return args


def main(argv=None):
    try:
        run_validation(parse_args(argv))
    except (ValidationError, OSError, csv.Error) as error:
        print(f"Dataset validation stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
