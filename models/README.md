# Local Model Staging

This directory is used temporarily to store trained models and tokenizers before publishing them to the Hugging Face Hub.

Model weights are intentionally excluded from Git tracking because they can be large.

## Expected workflow

1. Receive the final trained model and tokenizer.
2. Place the model artifacts inside this directory.
3. Validate the required files.
4. Publish the model to the Hugging Face Hub.
5. Verify the published repository.

The project standard base model is:

`bert-base-cased`
