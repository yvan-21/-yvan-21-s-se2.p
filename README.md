# Orchid leaf identification and health screening

This project is a Flask application for classifying orchid genera and screening
leaf images as `healthy`, `not_healthy`, or `unknown`. It uses a frozen
MobileNetV2 CNN feature extractor with dataset-trained species and health
classifiers. It does not diagnose a disease.

## Dataset layout

Keep the original images in `datasets/` and organize labels by folder:

```text
datasets/
  Cattleya/
    healthy/
    not healthy/
  Dendrobium/
    healthy/
    not healthy/
  Phalaenopsis/
    healthy/
    not healthy/
  Vanda/
    healthy/
    not healthy/
  image_groups.csv
```

`not_healthy` is also accepted as a folder name. Folder names are the training
labels. The current dataset has 158 images across these four genera, with an
uneven distribution; add more independently collected healthy examples,
particularly for underrepresented genera, before treating evaluation metrics
as dependable.

Training images and their group manifest are not included in this repository.
Provide your own labeled dataset and a verified `datasets/image_groups.csv`
before training. A small set of sample images used by the app's gallery is
included under `static/dataset/`.

The group manifest must contain `image_path`, `plant_id`, and `session_id`.
Give each photo the real plant identifier and, when known, the capture-session
identifier. Photos of the same plant or from the same capture session must
share the corresponding identifier. The trainer refuses missing or generated
per-image placeholder IDs so related photos cannot silently leak across
training, validation, and test partitions. Correct the manifest in a
spreadsheet, save it as UTF-8 CSV, and keep it at `datasets/image_groups.csv`.

## Train or retrain

Use the project's existing virtual environment from PowerShell:

```powershell
.\.venv\Scripts\python.exe .\train_model.py --scope all
```

The optional scope is `old`, `new`, or `all`; `all` is the default. The command
and the administrator dashboard use the same trainer. It validates the group
manifest, makes group-safe train/validation/test splits, augments only
development images, fits both classifiers, temperature-calibrates classifier
scores and calibrates uncertainty using held-out validation images, evaluates
on the test split, and saves the artifacts under `models/`. Training does not
alter the original dataset images.

The trainer saves:

- `models/species_feature_extractor_224.keras`
- `models/species_classifier_224.joblib`
- `models/health_classifier_224.joblib`
- `models/unknown_detection_224.json`
- `models/training_metrics.json`

The held-out validation data contains supported dataset labels only; this
project does not yet include a labeled set of non-orchid or unsupported-species
images. Feature-distance rejection and uncertainty sets can flag some
unfamiliar images, but their rejection rate on true out-of-distribution images
has not been measured. Calibration scores are validation-calibrated estimates,
not test accuracy or a guarantee for an individual photo.

## Start the application

Use Python 3.11 and install the dependencies from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Training images and `datasets/image_groups.csv` are not included; supply them
before training. Create a local `.env` file from the example and edit it with
your own session-signing key and admin password:

```powershell
Copy-Item .env.example .env
notepad .env
.\.venv\Scripts\python.exe .\generate_cert.py
.\.venv\Scripts\python.exe .\app.py
```

Replace both placeholder values in `.env`; keep the same `FLASK_SECRET_KEY`
and admin password for later runs. The admin username defaults to `admin`;
set `ADMIN_USERNAME` to change it. Admin login is disabled until
`ADMIN_PASSWORD` is set. Restart the app after changing `.env`. The file is
ignored by Git; the certificate, key, accounts, and scan records are also local
files and are not included in the repository. On a fresh setup, create a user
account through the sign-up form.

Open `https://localhost:5000` and accept the local development certificate
warning. The app loads saved model artifacts at startup; it does not retrain
automatically. If the calibration artifact is missing, the app shows the
classifier's dataset-label predictions as provisional, marks the scores as
uncalibrated, and warns that unsupported images may be forced into a supported
class. Complete a successful training run before relying on unknown-image
rejection.

## Test a new image

1. Sign in to the existing user account and open the scan/upload or camera page.
2. Upload or capture a clear, well-lit image showing one whole leaf.
3. Review the predicted genus and binary health screening label, the validation-
   calibrated model scores (when available), any image-quality warning, and the
   visible-cue explanation and care guidance. A calibrated score is not test
   accuracy or a diagnosis.
4. If the result is `Unknown / uncertain`, improve the image or consult the
   supported dataset labels. Do not treat an image-only screening result as a
   confirmed diagnosis.

Run the focused automated tests from the project root with:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_unknown_detection tests.test_image_group_manifest
.\.venv\Scripts\python.exe -m unittest tests.test_preprocess_dataset
```

## Main modules

- `app.py` — Flask routes, inference, augmentation, group-safe training, and
  evaluation.
- `unknown_detection.py` — pooled validation-calibrated prediction sets,
  feature-distance checks, and image-sharpness screening.
- `train_model.py` — command-line entry point for the same trainer used by the
  application.
- `preprocess_dataset.py` — separate dataset preview/preprocessing utility;
  it does not modify the originals. Its `preprocessing_stages/centered_square_species_view/`
  folder contains centered square crop previews corresponding to the second
  species-classification view.
- `templates/` — administrator, desktop/mobile upload, camera, and results
  pages.
- `tests/` — unit tests for data preparation, group splits, UI output, and
  uncertainty calibration.
