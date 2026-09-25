---
language:
- en
license: apache-2.0
tags:
- question-answering
- bert
- pytorch
- squad
- extractive-qa
datasets:
- rajpurkar/squad
metrics:
- f1
- exact_match
pipeline_tag: question-answering
widget:
- text: "Where is the University of Notre Dame located?"
  context: "The University of Notre Dame is a private Catholic research university in Notre Dame, Indiana, outside the city of South Bend."
---

# 🧠 BERT-Base Cased for Extractive Question Answering (SQuAD v1.1)

This model is a fine-tuned version of [bert-base-cased](https://huggingface.co/bert-base-cased) on the full [SQuAD v1.1](https://huggingface.co/datasets/rajpurkar/squad) dataset.

## 📊 Model Performance

Evaluated on the full SQuAD v1.1 validation set (10,570 examples):

| Metric | Score | Official BERT-Base Paper |
| :--- | :---: | :---: |
| **Exact Match (EM)** | **80.99%** | 81.50% |
| **F1-Score** | **88.21%** | 88.50% |

## 🚀 Hyperparameters & Training Setup

- **Base Model:** `bert-base-cased`
- **Training Samples:** 87,599 (100% of SQuAD v1.1)
- **Epochs:** 3
- **Optimizer:** AdamW (weight_decay: 0.01)
- **Learning Rates:** QA Head `3e-4` | BERT Encoder `3e-5`
- **Scheduler:** Cosine Annealing with 10% Linear Warmup
- **Precision:** PyTorch Mixed Precision (AMP FP16)
- **Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU

## 💻 Usage Example

```python
import torch
from transformers import BertForQuestionAnswering, AutoTokenizer

model_id = "RusselKuAguilar/bert-base-cased-squad-extractive-qa"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = BertForQuestionAnswering.from_pretrained(model_id)
model.eval()

context = "Python was created in the late 1980s by Guido van Rossum at CWI in the Netherlands."
question = "Who created Python?"

inputs = tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits)
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx : end_idx + 1])
)
print("Answer:", answer) # -> Guido van Rossum
```
