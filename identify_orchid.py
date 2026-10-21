
import argparse
import os
import requests
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = (
    PROJECT_DIR / "datasets" / "Vanda" / "healthy" / "IMG20261011093358.jpg"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify an orchid photo with the Pl@ntNet API."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=str(DEFAULT_IMAGE_PATH),
        help=(
            "Image to identify. Defaults to a Vanda image in datasets. "
            "Relative paths are resolved from the current directory."
        ),
    )
    return parser.parse_args()


def main():
    image_path = Path(parse_args().image).expanduser()
    api_key = os.getenv("PLANTNET_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set your PLANTNET_API_KEY environment variable.")

    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    url = "https://my-api.plantnet.org/v2/identify/all"
    try:
        with image_path.open("rb") as image:
            response = requests.post(
                url,
                params={"api-key": api_key},
                files={"images": (image_path.name, image, "image/jpeg")},
                data={"organs": "leaf"},
                timeout=60,
            )
    except requests.RequestException as error:
        raise SystemExit(
            f"API request failed ({type(error).__name__}); check network access."
        ) from None

    if not response.ok:
        raise SystemExit(
            f"Pl@ntNet returned HTTP {response.status_code}; check API key access "
            "and quota. Response details omitted."
        )

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        raise SystemExit("Pl@ntNet returned an invalid JSON response.") from None

    results = data.get("results", [])
    if not isinstance(results, list) or not results:
        print("No likely species were returned.")
        return

    print(f"Image: {image_path}")
    print("\nPossible species:")
    for result in results[:5]:
        species = result.get("species", {})
        name = species.get("scientificNameWithoutAuthor", "Unknown")
        score = result.get("score", 0)
        print(f"{name} — score: {score:.2%}")


if __name__ == "__main__":
    main()
