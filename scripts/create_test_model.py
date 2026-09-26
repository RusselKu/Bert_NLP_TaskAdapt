#!/usr/bin/env python3

"""
Create a tiny random BERT model to test the deployment pipeline.

IMPORTANT:
This is NOT one of the final U2T01 models.
It is only a smoke-test artifact for validating the uploader.
"""

from pathlib import Path

from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
)


OUTPUT_DIR = Path("models/uploader-smoke-test")

PROJECT_TOKENIZER = "google-bert/bert-base-cased"


def main():
    print("Creating Hugging Face deployment smoke-test model...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load the tokenizer agreed upon by the project
    # ---------------------------------------------------------

    print(
        f"Loading tokenizer: {PROJECT_TOKENIZER}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_TOKENIZER
    )

    # ---------------------------------------------------------
    # Create a very small random BERT
    # ---------------------------------------------------------

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        num_labels=2,
    )

    model = BertForSequenceClassification(
        config
    )

    # ---------------------------------------------------------
    # Save model and tokenizer
    # ---------------------------------------------------------

    model.save_pretrained(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    # ---------------------------------------------------------
    # Create test Model Card
    # ---------------------------------------------------------

    model_card = """---
language:
- en
library_name: transformers
tags:
- bert
- testing
- deployment
---

# BERT Uploader Smoke Test

This repository contains a tiny randomly initialized BERT model.

It was created only to validate the Hugging Face deployment workflow
for the U2T01: Adapting BERT for NLP Tasks project.

## Project Standard

The project uses:

`google-bert/bert-base-cased`

The tokenizer used by this smoke test comes from that model.

## Intended Use

Deployment and uploader testing only.

## Limitations

This model is randomly initialized.

It has not been trained on an NLP dataset and therefore has no
meaningful predictive capability.

It must not be reported as one of the final U2T01 models.
"""

    readme_path = OUTPUT_DIR / "README.md"

    readme_path.write_text(
        model_card,
        encoding="utf-8",
    )

    print("")
    print("✓ Test model created successfully")
    print(f"✓ Location: {OUTPUT_DIR}")
    print(f"✓ Tokenizer: {PROJECT_TOKENIZER}")
    print("✓ Model Card created")
    print("")
    print(
        "This model is for deployment testing only."
    )


if __name__ == "__main__":
    main()
