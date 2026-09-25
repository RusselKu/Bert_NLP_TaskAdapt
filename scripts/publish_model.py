#!/usr/bin/env python3

"""
Generic Hugging Face uploader for the U2T01 BERT adaptation project.

This script:
1. Validates a local Hugging Face Transformers model directory.
2. Checks that the model architecture is BERT.
3. Checks that model weights exist.
4. Checks that tokenizer artifacts exist.
5. Checks whether a Model Card (README.md) exists.
6. Verifies Hugging Face authentication.
7. Creates the target Hugging Face repository if necessary.
8. Uploads the complete model directory.

Project standard:
    google-bert/bert-base-cased
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi


PROJECT_BASE_MODEL = "google-bert/bert-base-cased"


def validate_model_folder(model_path: Path) -> None:
    """Validate the minimum expected Hugging Face model artifacts."""

    print(f"Validating: {model_path}")

    # ---------------------------------------------------------
    # 1. Check directory
    # ---------------------------------------------------------

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory does not exist: {model_path}"
        )

    if not model_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {model_path}"
        )

    files = {
        item.name
        for item in model_path.iterdir()
        if item.is_file()
    }

    # ---------------------------------------------------------
    # 2. Validate config.json
    # ---------------------------------------------------------

    config_path = model_path / "config.json"

    if "config.json" not in files:
        raise ValueError(
            "Missing config.json. "
            "The model should be exported using model.save_pretrained()."
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "config.json exists but is not valid JSON."
        ) from exc

    model_type = config.get("model_type")

    if model_type != "bert":
        raise ValueError(
            f"Expected a BERT model, but config.json reports "
            f"model_type={model_type!r}."
        )

    # ---------------------------------------------------------
    # 3. Validate model weights
    # ---------------------------------------------------------

    weight_files = {
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }

    if not files.intersection(weight_files):
        raise ValueError(
            "Model weights not found. Expected one of: "
            "model.safetensors, pytorch_model.bin, "
            "model.safetensors.index.json, "
            "or pytorch_model.bin.index.json."
        )

    # ---------------------------------------------------------
    # 4. Validate tokenizer
    # ---------------------------------------------------------

    tokenizer_core_files = {
        "tokenizer.json",
        "vocab.txt",
    }

    if not files.intersection(tokenizer_core_files):
        raise ValueError(
            "Tokenizer vocabulary was not detected. "
            "Expected tokenizer.json or vocab.txt. "
            "Make sure tokenizer.save_pretrained() was executed."
        )

    if "tokenizer_config.json" not in files:
        print(
            "WARNING: tokenizer_config.json was not found. "
            "Verify that the tokenizer was exported correctly."
        )

    # ---------------------------------------------------------
    # 5. Check Model Card
    # ---------------------------------------------------------

    if "README.md" not in files:
        print(
            "WARNING: README.md Model Card was not found. "
            "Final submitted models must include one."
        )
    else:
        print("✓ Model Card detected")

    # ---------------------------------------------------------
    # Validation summary
    # ---------------------------------------------------------

    print("✓ config.json found and valid")
    print("✓ BERT architecture detected")
    print("✓ model weights detected")
    print("✓ tokenizer artifacts detected")
    print(f"✓ Project base-model standard: {PROJECT_BASE_MODEL}")
    print("✓ Local validation passed")


def get_hf_identity(api: HfApi) -> str:
    """Verify Hugging Face authentication and return username."""

    try:
        user_info = api.whoami()
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face authentication failed. "
            "Run 'hf auth login' before publishing."
        ) from exc

    username = user_info.get("name")

    if not username:
        raise RuntimeError(
            "Could not determine the authenticated Hugging Face username."
        )

    return username


def publish_model(
    model_path: Path,
    repo_id: str,
    private: bool,
) -> None:
    """Create the Hub repository and upload the local model folder."""

    api = HfApi()

    username = get_hf_identity(api)

    print(f"✓ Authenticated as: {username}")
    print(f"Target repository: {repo_id}")
    print(f"Visibility: {'private' if private else 'public'}")

    print("Creating/checking Hugging Face repository...")

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    print("✓ Repository ready")
    print("Uploading model artifacts...")

    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=(
            "Upload trained BERT model, tokenizer and Model Card"
        ),
    )

    print("✓ Upload completed successfully")
    print(f"✓ Repository URL: https://huggingface.co/{repo_id}")


def parse_arguments():
    """Define command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate and publish a trained BERT model "
            "and tokenizer to the Hugging Face Hub."
        )
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help=(
            "Local directory containing the exported model "
            "and tokenizer."
        ),
    )

    parser.add_argument(
        "--repo-id",
        required=True,
        help=(
            "Target Hugging Face repository. "
            "Example: DamianNv/bert-agnews-topic-classification"
        ),
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face repository as private.",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate local model artifacts without "
            "creating or uploading a repository."
        ),
    )

    return parser.parse_args()


def main():
    """Main execution flow."""

    args = parse_arguments()

    model_path = Path(
        args.model_path
    ).expanduser().resolve()

    try:
        validate_model_folder(model_path)

        if args.validate_only:
            print("✓ Validation-only mode completed")
            return

        publish_model(
            model_path=model_path,
            repo_id=args.repo_id,
            private=args.private,
        )

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
