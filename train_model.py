"""Command-line entry point for the application's group-safe model trainer."""

import argparse
import os

import app as orchid_app


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the Orchid application's classifiers on the labeled datasets/ images."
    )
    parser.add_argument(
        "--scope",
        choices=("old", "new", "all"),
        default="all",
        help="Dataset image scope to train on (default: all).",
    )
    args = parser.parse_args()

    orchid_app.orchid_ai.species_classes = orchid_app.orchid_ai.get_available_species()
    image_paths = orchid_app.get_training_image_paths(args.scope)
    if len(image_paths) < 10:
        parser.error(
            f"Scope {args.scope!r} has only {len(image_paths)} image(s); at least 10 are required."
        )

    try:
        orchid_app.repair_missing_group_manifest_ids(
            orchid_app.get_dataset_image_paths()
        )
    except ValueError as error:
        parser.error(f"Could not assign group IDs automatically: {error}")

    image_group_ids, group_error = orchid_app.load_image_group_assignments(
        image_paths,
        allow_extra_paths=True,
    )
    if group_error:
        parser.error(
            f"{group_error} Add plant/session IDs in datasets/image_groups.csv before training."
        )

    species_labels, health_labels = orchid_app.get_dataset_labels_for_paths(image_paths)
    group_ids = [image_group_ids[path] for path in image_paths]
    try:
        orchid_app.make_grouped_train_validation_test_split(
            species_labels,
            group_ids,
            "species classification",
        )
        orchid_app.make_grouped_train_validation_test_split(
            health_labels,
            group_ids,
            "health classification",
        )
    except ValueError as error:
        parser.error(str(error))

    dataset_fingerprint = orchid_app.get_dataset_fingerprint(image_paths, image_group_ids)
    if not orchid_app.training_lock.acquire(blocking=False):
        parser.error("Model training is already running in this process.")

    try:
        orchid_app.training_progress.update({
            "status": "training",
            "percentage": 1,
            "current_stage": "Initializing",
            "message": "Initializing model training...",
            "error": None,
        })
        orchid_app.train_model_background(
            image_paths,
            image_group_ids,
            dataset_fingerprint,
            args.scope,
        )
    finally:
        orchid_app.training_lock.release()

    if orchid_app.training_progress.get("status") != "completed":
        print(orchid_app.training_progress.get("message", "Model training failed."))
        return 1

    metrics = orchid_app.load_app_document(
        "training_metrics",
        os.path.join(orchid_app.BASE_DIR, "models", "training_metrics.json"),
        {},
    )
    print(
        "Training completed: "
        f"{metrics.get('total_images', len(image_paths))} images; "
        f"species test accuracy={metrics.get('accuracy', 'unavailable')}; "
        f"health test accuracy={metrics.get('health_accuracy', 'unavailable')}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
