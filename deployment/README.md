# Hugging Face Deployment Guide

This directory documents the deployment workflow for the U2T01:
**Adapting BERT for NLP Tasks** project.

## Project Standard

The team uses:

`google-bert/bert-base-cased`

as the common BERT-base model family.

## Deployment Components

The deployment workflow includes:

- `scripts/publish_model.py`
  - Validates local model artifacts.
  - Checks BERT architecture.
  - Checks model weights.
  - Checks tokenizer artifacts.
  - Checks for a Model Card.
  - Creates or reuses a Hugging Face repository.
  - Uploads the model directory.

- `scripts/verify_hub_model.py`
  - Verifies that a published repository is accessible.
  - Checks remote artifacts.
  - Loads the configuration.
  - Loads the tokenizer.
  - Detects the task-specific head.
  - Loads the published model.

- `scripts/create_test_model.py`
  - Creates a tiny random BERT model used only for deployment testing.
  - It must not be reported as an assignment result.

- `templates/model_card_template.md`
  - Template for the Model Card required for each final model.

- `notebooks/deployment/huggingface_publish.ipynb`
  - Google Colab/Jupyter workflow for validation, publication and verification.

## Tested Environment

The deployment workflow was tested successfully with:

| Component | Version |
|---|---|
| Python | 3.11.16 |
| PyTorch | 2.14.0 |
| Transformers | 5.17.0 |
| Hugging Face Hub | 1.33.0 |
| Jupyter | 1.1.1 |
| ipykernel | 7.3.0 |

The local PyTorch installation used a CUDA-specific build. The
`requirements-deployment.txt` file intentionally avoids pinning a
CUDA-specific PyTorch build so the deployment workflow can also run
on Google Colab.
## Local Setup

Create and activate the deployment environment:

```bash
conda create -n bert-nlp python=3.11 -y
conda activate bert-nlp
pip install -r requirements-deployment.txt
