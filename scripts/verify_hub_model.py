#!/usr/bin/env python3

"""
Verify a BERT model published to the Hugging Face Hub.

Checks:
1. Repository accessibility.
2. Required remote artifacts.
3. Transformers configuration.
4. BERT architecture.
5. Tokenizer loading.
6. Tokenizer test.
7. Correct task-specific model head loading.

U2T01 - Adapting BERT for NLP Tasks
"""

import argparse
import sys

from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


def load_task_model(repo_id: str, config):
    """
    Load the model using the correct task-specific AutoModel class.

    Supported U2T01 architectures:
    - Sequence Classification
    - Token Classification (NER/POS)
    - Question Answering
    """

    architectures = getattr(
        config,
        "architectures",
        None,
    ) or []

    architecture = (
        architectures[0]
        if architectures
        else ""
    )

    print(
        f"Detected architecture from config: "
        f"{architecture or 'not specified'}"
    )

    if "ForSequenceClassification" in architecture:
        print("Detected task head: Sequence Classification")

        model = (
            AutoModelForSequenceClassification
            .from_pretrained(repo_id)
        )

        task = "sequence-classification"

    elif "ForTokenClassification" in architecture:
        print("Detected task head: Token Classification")

        model = (
            AutoModelForTokenClassification
            .from_pretrained(repo_id)
        )

        task = "token-classification"

    elif "ForQuestionAnswering" in architecture:
        print("Detected task head: Question Answering")

        model = (
            AutoModelForQuestionAnswering
            .from_pretrained(repo_id)
        )

        task = "question-answering"

    else:
        print(
            "WARNING: Task-specific architecture could not "
            "be inferred. Falling back to AutoModel."
        )

        model = AutoModel.from_pretrained(
            repo_id
        )

        task = "base-model"

    return model, task


def verify_repository(repo_id: str) -> None:
    """Verify a model repository published on Hugging Face."""

    api = HfApi()

    print("")
    print("========================================")
    print("HUGGING FACE MODEL VERIFICATION")
    print("========================================")
    print(f"Repository: {repo_id}")
    print("")

    # ---------------------------------------------------------
    # 1. Repository accessibility
    # ---------------------------------------------------------

    try:
        repo_info = api.model_info(
            repo_id
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not access repository: {repo_id}"
        ) from exc

    print("✓ Repository is accessible")

    # ---------------------------------------------------------
    # 2. Inspect remote files
    # ---------------------------------------------------------

    files = {
        sibling.rfilename
        for sibling in repo_info.siblings
    }

    print(
        f"✓ Repository contains {len(files)} files"
    )

    if "config.json" not in files:
        raise ValueError(
            "Remote repository is missing config.json"
        )

    weight_candidates = {
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }

    if not files.intersection(
        weight_candidates
    ):
        raise ValueError(
            "Remote repository does not contain model weights"
        )

    tokenizer_candidates = {
        "tokenizer.json",
        "vocab.txt",
    }

    if not files.intersection(
        tokenizer_candidates
    ):
        raise ValueError(
            "Remote repository does not contain tokenizer files"
        )

    if "README.md" not in files:
        print(
            "WARNING: Model Card README.md was not found"
        )
    else:
        print("✓ Model Card detected")

    print("✓ Model weights detected")
    print("✓ Tokenizer artifacts detected")

    # ---------------------------------------------------------
    # 3. Load configuration
    # ---------------------------------------------------------

    config = AutoConfig.from_pretrained(
        repo_id
    )

    print("✓ Configuration downloaded")

    if config.model_type != "bert":
        raise ValueError(
            f"Expected BERT architecture, "
            f"got {config.model_type!r}"
        )

    print("✓ BERT architecture confirmed")

    # ---------------------------------------------------------
    # 4. Load tokenizer
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id
    )

    print(
        f"✓ Tokenizer loaded: "
        f"{tokenizer.__class__.__name__}"
    )

    sample_text = (
        "This is a Hugging Face deployment test."
    )

    encoded = tokenizer(
        sample_text,
        return_tensors="pt",
    )

    if "input_ids" not in encoded:
        raise ValueError(
            "Tokenizer did not produce input_ids"
        )

    print(
        f"✓ Tokenizer test passed "
        f"(shape={tuple(encoded['input_ids'].shape)})"
    )

    # ---------------------------------------------------------
    # 5. Load the correct task-specific model
    # ---------------------------------------------------------

    model, task = load_task_model(
        repo_id,
        config,
    )

    print(
        f"✓ Model loaded: "
        f"{model.__class__.__name__}"
    )

    print(
        f"✓ Model type: "
        f"{model.config.model_type}"
    )

    print(
        f"✓ Task family: {task}"
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("")
    print("========================================")
    print("VERIFICATION PASSED")
    print("========================================")
    print(f"Repository: {repo_id}")
    print(f"Architecture: {config.model_type}")
    print(f"Task family: {task}")
    print(
        f"Tokenizer: "
        f"{tokenizer.__class__.__name__}"
    )
    print(
        f"Model class: "
        f"{model.__class__.__name__}"
    )
    print("")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Verify a BERT model repository "
            "published to Hugging Face."
        )
    )

    parser.add_argument(
        "--repo-id",
        required=True,
        help=(
            "Hugging Face repository ID. "
            "Example: "
            "DamianNv/bert-uploader-smoke-test"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        verify_repository(
            args.repo_id
        )

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
