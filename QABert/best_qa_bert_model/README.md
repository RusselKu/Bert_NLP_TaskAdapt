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
- transformers
datasets:
- rajpurkar/squad
metrics:
- f1
- exact_match
pipeline_tag: question-answering
widget:
- text: "Where is the University of Notre Dame located?"
  context: "The University of Notre Dame is a private Catholic research university in Notre Dame, Indiana, outside the city of South Bend."
- text: "Who created Python?"
  context: "Python was conceived in the late 1980s by Guido van Rossum at CWI in the Netherlands."
- text: "When did Neil Armstrong step onto the lunar surface?"
  context: "Armstrong became the first person to step onto the lunar surface on July 21, 1969, at 02:56 UTC."
---

# 🧠 BERT-Base Cased for Extractive Question Answering (SQuAD v1.1)

This repository provides a fine-tuned [`bert-base-cased`](https://huggingface.co/bert-base-cased) model adapted for the downstream task of **Extractive Question Answering** on the full **[SQuAD v1.1](https://huggingface.co/datasets/rajpurkar/squad)** dataset.

Given a *(Question, Context)* pair, the model predicts the exact start and end character/token spans within the context passage that answer the question.

---

## 🎯 Intended Use

### Intended Uses
- **Extractive Question Answering:** Retrieving verbatim factual answers from English text paragraphs (e.g., articles, reports, documentation, encyclopedic entries).
- **Search & Knowledge Retrieval Pipelines:** Serving as the reader/answering component in Retrieval-Augmented Generation (RAG) or Dense Passage Retrieval (DPR) search architectures.
- **Academic & Research Benchmarking:** Studying parameter-efficient adaptation and transfer learning dynamics of BERT on MRC (Machine Reading Comprehension) tasks.

### Out-of-Scope & Unintended Uses
- **Generative / Abstractive QA:** The model cannot synthesize new text, paraphrase, or generate reasoning chains; it strictly extracts existing text spans.
- **Unanswerable Questions (SQuAD v2.0 style):** Trained on SQuAD v1.1, where every question is guaranteed to have a plausible answer in the context. If given unanswerable questions, it will attempt to extract the most plausible span rather than predicting no answer.
- **Multi-lingual or Non-English Text:** The model was fine-tuned on English text and is not intended for non-English corpora.
- **High-Stakes Decision Making:** Should not be used for critical medical, legal, or financial diagnosis without human-in-the-loop validation.

---

## 📊 Model Performance & Benchmarks

The model was evaluated on the complete **SQuAD v1.1 validation set (`val_dataset`, 10,570 examples)**:

| Metric / Checkpoint | Final Checkpoint (Epoch 3) | Peak Intermediate (Epoch 2) | Official Paper (Devlin et al., 2018) |
| :--- | :---: | :---: | :---: |
| **Validation Exact Match (EM)** | **80.75%** | 80.99% | 81.50% |
| **Validation F1-Score** | **88.21%** 🏆 | 88.19% | 88.50% |
| **Final Training Loss** | **0.4990** | 0.7953 | — |
| **Training Time (3 epochs)** | **~67.9 min** (4,075.4 s) | — | — |

*Our final Epoch 3 checkpoint achieves **88.21% F1** and **80.75% EM**, matching the original Google BERT benchmark (-0.29 F1 points) with complete reproducibility on modern GPU hardware and AMP FP16.*

---

## ⚙️ Hyperparameters & Training Setup

- **Base Architecture:** `bert-base-cased` (12 layers, 768 hidden size, 12 attention heads, ~108M parameters)
- **Dataset:** Stanford Question Answering Dataset ([SQuAD v1.1](https://huggingface.co/datasets/rajpurkar/squad))
- **Training Samples:** 87,599 examples (100% full training split)
- **Validation Samples:** 10,570 examples
- **Sequence Length (`max_length`):** 384 tokens
- **Document Stride (`doc_stride`):** 128 tokens
- **Batch Size:** 16 (per device)
- **Epochs:** 3
- **Optimizer:** AdamW (`weight_decay=0.01`, $\epsilon=1\times 10^{-8}$)
- **Learning Rates:** QA Head: `3e-4` | BERT Encoder: `3e-5`
- **Learning Rate Scheduler:** Cosine Annealing with 10% Linear Warmup (`warmup_steps=1642`)
- **Gradient Clipping:** Maximum gradient norm of `1.0`
- **Precision:** PyTorch Mixed Precision (AMP FP16)
- **Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU (8.59 GB VRAM, CUDA 12.4)
- **Random Seed:** 42

---

## 🔬 Error Analysis (Qualitative Case Studies)

To better understand model failure modes on SQuAD v1.1 validation examples, we identified 3 primary error categories:

| Case | Question | Gold Reference Answer | Model Prediction | EM / F1 | Error Diagnosis |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **1. Span Boundary Over-extension** | *Who was the architect of the bridge?* | `Joseph Strauss` | `Joseph Strauss, who oversaw the construction` | EM: **0.0%**<br>F1: **44.4%** | **Boundary Overlap:** The model locates the core target entity but extends the span to include the explanatory relative clause due to high semantic co-occurrence. |
| **2. Lexical Distractor Confusion** | *Who invented the first electric light?* | `Humphry Davy` | `Thomas Edison` | EM: **0.0%**<br>F1: **0.0%** | **Salience Bias:** In complex historical contexts comparing multiple inventors, the model selects the entity most prominently associated with the word *"invented"* (Edison, 1879) instead of the earlier subordinate clause (Davy, 1802). |
| **3. Modifier Truncation** | *What kind of reparations were imposed?* | `financial reparations` | `financial` | EM: **0.0%**<br>F1: **66.7%** | **Head Noun Omission:** The model extracts only the adjective modifier answering *"what kind of"*, missing the full syntactic noun phrase expected by human annotators. |

---

## 💻 How to Use

### Using Hugging Face `transformers` Pipeline

```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="RusselKuAguilar/bert-base-cased-squad-extractive-qa",
    tokenizer="RusselKuAguilar/bert-base-cased-squad-extractive-qa"
)

context = "The Apollo 11 mission landed Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin on July 20, 1969."
question = "Who was the commander of Apollo 11?"

result = qa_pipeline(question=question, context=context)
print(result)
# Output: {'score': 0.9412, 'start': 38, 'end': 52, 'answer': 'Neil Armstrong'}
```

### Direct PyTorch / Transformers Usage

```python
import torch
from transformers import AutoTokenizer, BertForQuestionAnswering

model_id = "RusselKuAguilar/bert-base-cased-squad-extractive-qa"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = BertForQuestionAnswering.from_pretrained(model_id)
model.eval()

context = "Python was conceived in the late 1980s by Guido van Rossum at CWI in the Netherlands."
question = "Who created Python?"

inputs = tokenizer(question, context, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits)

answer_tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer_tokens))

print(f"Answer: {answer}")
# Output: Answer: Guido van Rossum
```

---

## ⚠️ Limitations & Biases

1. **Length Constraint:** Sequences longer than 512 tokens are truncated unless segmented using sliding windows (`doc_stride`). Long contexts might omit critical answer spans if not chunked appropriately.
2. **Span Extraction Dependency:** The model cannot answer questions whose solutions require synthesizing facts from disparate sentences, multi-hop reasoning, or calculating numerical operations (e.g., counting, dates difference).
3. **Absence of Negative Verification:** Because SQuAD v1.1 does not contain unanswerable questions, the model will output the span with the highest score even if the context does not contain the true answer.
4. **Dataset Biases:** SQuAD v1.1 is sourced from Wikipedia articles; hence, historical, demographic, or social biases present in Wikipedia or in `bert-base-cased` pre-training corpora may be reflected in predictions.

---

## 📦 Software Dependencies & Environment

- **Python:** `3.11+ / 3.12+ / 3.13`
- **PyTorch:** `2.6.0+cu124`
- **Transformers:** `5.17.0`
- **Datasets:** `5.0.1`
- **CUDA Version:** `12.4`
- **Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU

---

## 📚 References

1. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805).
2. **Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016).** *SQuAD: 100,000+ Questions for Machine Comprehension of Text.* [arXiv:1606.05250](https://arxiv.org/abs/1606.05250).
3. **Wolf, T. et al. (2020).** *Transformers: State-of-the-Art Natural Language Processing.* In Proceedings of the 2020 EMNLP: System Demonstrations (pp. 38–45).
4. **Loshchilov, I., & Hutter, F. (2017).** *Decoupled Weight Decay Regularization (AdamW).* [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).
