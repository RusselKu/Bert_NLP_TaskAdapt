# Team Model Handoff Checklist

Complete this checklist before sending a final model to Damian for deployment.

## General Information

- Responsible member:
- Task:
- Dataset:
- Base model: `google-bert/bert-base-cased`
- Final adaptation method:
- Alternative method tested:

## Model Artifacts

- [ ] `config.json`
- [ ] Model weights (`model.safetensors` or `pytorch_model.bin`)
- [ ] Tokenizer files
- [ ] Model Card / README
- [ ] Final model loads correctly

## Training Configuration

- Random seed:
- Epochs:
- Batch size:
- Head learning rate:
- Encoder learning rate:
- Trainable parameters:
- Total parameters:

## Training Metrics

- Training loss:
- Evaluation loss:
- Score per epoch:
- Training time:
- Seconds per epoch / step:

## Final Evaluation

- Main metric:
- Final score:
- Additional metrics:

## Experimental Comparison

- Method A:
- Result:
- Method B:
- Result:
- Selected method:
- Selection justification:

## Documentation

- [ ] Training data documented
- [ ] Metrics documented
- [ ] Intended use documented
- [ ] Limitations documented
- [ ] References documented

## Handoff

Provide one of:

- [ ] Local exported model directory
- [ ] Hugging Face model URL

Hugging Face URL:

`________________________________________`

## Deployment Result

To be completed by Damian:

- [ ] Local validation passed
- [ ] Hugging Face repository accessible
- [ ] Model Card detected
- [ ] Model weights detected
- [ ] Tokenizer detected
- [ ] Correct BERT architecture
- [ ] Correct task head
- [ ] Remote model loads successfully
- [ ] `VERIFICATION PASSED`
