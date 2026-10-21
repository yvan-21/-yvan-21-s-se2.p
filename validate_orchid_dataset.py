
import os
import csv
import time
import argparse
from pathlib import Path

import requests

API_URL = "https://my-api.plantnet.org/v2/identify/all"
MAX_REQUEST_ATTEMPTS = 3
REQUEST_TIMEOUT = (20, 90)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}

# Review threshold only; this is not a calibrated probability.
MIN_SCORE = 0.10


def identify_orchid(image_path, api_key, organ="leaf"):
    """Send one image to Pl@ntNet, retrying temporary network failures."""
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            # Reopen the image for each attempt because requests consumes the stream.
            with image_path.open("rb") as image_file:
                response = requests.post(
                    API_URL,
                    params={"api-key": api_key},
                    files=[
                        ("images", (image_path.name, image_file)),
                        ("organs", (None, organ)),
                    ],
                    timeout=REQUEST_TIMEOUT,
                )
            break
        except (requests.Timeout, requests.ConnectionError) as error:
            if attempt == MAX_REQUEST_ATTEMPTS:
                raise RuntimeError(
                    "Pl@ntNet request failed after "
                    f"{MAX_REQUEST_ATTEMPTS} attempts "
                    f"({type(error).__name__})."
                ) from None

            delay = 2 * attempt
            print(
                f"  Temporary {type(error).__name__}; "
                f"retrying in {delay}s ({attempt}/{MAX_REQUEST_ATTEMPTS})..."
            )
            time.sleep(delay)

    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code}: {response.text[:300]}"
        )

    return response.json()


def get_predictions(data):
    """Extract the five highest-ranked species."""
    predictions = []

    for result in data.get("results", []):
        species = result.get("species", {})

        name = (
            species.get("scientificNameWithoutAuthor")
            or species.get("scientificName")
            or "Unknown"
        )

        score = result.get("score")
        predictions.append((name, score))

    top_name = predictions[0][0] if predictions else ""
    top_score = predictions[0][1] if predictions else None

    top_five = " | ".join(
        f"{name} ({score:.2%})"
        if isinstance(score, (int, float))
        else name
        for name, score in predictions[:5]
    )

    return top_name, top_score, top_five


def main():
    parser = argparse.ArgumentParser(
        description="Validate orchid dataset labels using Pl@ntNet."
    )

    parser.add_argument(
        "--dataset",
        default="datasets",
        help="Dataset directory (default: datasets)",
    )

    parser.add_argument(
        "--output",
        default="orchid_validation_report.csv",
        help="CSV report filename",
    )

    parser.add_argument(
        "--organ",
        default="leaf",
        choices=["leaf", "flower", "fruit", "bark", "auto"],
        help="Plant organ in the images (default: leaf)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum images to test; 0 means all",
    )

    args = parser.parse_args()

    api_key = os.getenv("PLANTNET_API_KEY", "").strip()

    if not api_key:
        raise SystemExit(
            'API key missing. Run: $env:PLANTNET_API_KEY="YOUR_KEY"'
        )

    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()

    if not dataset.is_dir():
        raise SystemExit(f"Dataset folder not found: {dataset}")

    images = sorted(
        p for p in dataset.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and p.resolve() != output
    )

    if args.limit > 0:
        images = images[:args.limit]

    if not images:
        raise SystemExit(f"No images found in {dataset}")

    columns = [
        "image_path",
        "species_label",
        "health_label",
        "top_predicted_species",
        "top_score",
        "top_5_predictions",
        "species_label_match",
        "validation_status",
        "error",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset}")
    print(f"Images found: {len(images)}")
    print("Starting Pl@ntNet identification...\n")

    with output.open(
        "w", newline="", encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()

        for index, image in enumerate(images, start=1):
            relative = image.relative_to(dataset)
            parts = relative.parts

            # Expected layout: datasets/species/health/image.jpg
            species_label = parts[0] if len(parts) >= 2 else ""
            health_label = parts[1] if len(parts) >= 3 else ""

            row = {
                "image_path": str(relative),
                "species_label": species_label,
                "health_label": health_label,
                "top_predicted_species": "",
                "top_score": "",
                "top_5_predictions": "",
                "species_label_match": "Unknown",
                "validation_status": "",
                "error": "",
            }

            print(f"[{index}/{len(images)}] {relative}")

            try:
                data = identify_orchid(
                    image, api_key, args.organ
                )

                top_name, top_score, top_five = get_predictions(data)

                row["top_predicted_species"] = top_name
                row["top_score"] = (
                    f"{top_score:.6f}"
                    if isinstance(top_score, (int, float))
                    else ""
                )
                row["top_5_predictions"] = top_five

                if not top_name:
                    row["validation_status"] = (
                        "Manual review: no species result"
                    )
                    row["species_label_match"] = "No prediction"

                else:
                    # This is only a rough name check, not definitive validation.
                    match = (
                        species_label.casefold()
                        in top_name.casefold()
                    )

                    row["species_label_match"] = (
                        "Possible match" if match else "No direct match"
                    )

                    if (
                        isinstance(top_score, (int, float))
                        and top_score < MIN_SCORE
                    ):
                        row["validation_status"] = (
                            "Manual review: low top score"
                        )
                    elif match:
                        row["validation_status"] = (
                            "Provisional agreement; verify manually"
                        )
                    else:
                        row["validation_status"] = (
                            "Manual review: label differs from top result"
                        )

                print(
                    f"  Top result: {top_name or 'None'} "
                    f"({top_score if top_score is not None else 'N/A'})"
                )

            except (requests.RequestException, RuntimeError, ValueError, OSError) as exc:
                row["validation_status"] = "Identification error"
                row["error"] = str(exc).replace("\n", " ")[:500]
                print(f"  ERROR: {row['error']}")

            writer.writerow(row)
            csv_file.flush()

            # A short pause between requests; check your API's usage limits.
            if index < len(images):
                time.sleep(1)

    print("\nDataset validation finished.")
    print(f"CSV report saved to: {output}")


if __name__ == "__main__":
    main()
