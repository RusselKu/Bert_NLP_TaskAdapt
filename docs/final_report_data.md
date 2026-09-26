# U2T01 — Consolidated Experimental Results

## 1. Named Entity Recognition — Jonathan

Dataset: CoNLL-2003
Base model: bert-base-cased
Primary metric: strict entity-level F1

| Method | Validation F1 | Test F1 | Test Precision | Test Recall | Test Accuracy | Training Time |
|---|---:|---:|---:|---:|---:|---:|
| Partial Fine-Tuning | 91.71% | 88.79% | 89.03% | 88.54% | 97.75% | 115.63 s |
| Full Fine-Tuning | 95.11% | 91.54% | 91.55% | 91.52% | 98.28% | 358.54 s |

Selected method: Full Fine-Tuning

Reason:
Full achieved +2.75 percentage points of test F1, at approximately
3.1x the training time and 3.2x the peak VRAM.

Status: FROZEN.


## 2. Part-of-Speech Tagging — Rivaldo

Dataset: Universal Dependencies English EWT
Base model: bert-base-cased
Primary metric: Macro F1

| Method | Test Accuracy | Macro F1 | Precision | Recall | Test Loss |
|---|---:|---:|---:|---:|---:|
| Partial Fine-Tuning | 95.43% | 89.09% | 89.97% | 88.65% | 0.1520 |
| Full Fine-Tuning | 97.42% | 94.46% | 95.36% | 93.84% | 0.1134 |

Selected method: Full Fine-Tuning

Learning rates:
- Classification head: 1e-3
- Pretrained encoder: 2e-5

Reason:
Full obtained +5.37 percentage points of Macro F1.

Important qualification:
The experiments differ not only in the number of trainable encoder layers.
Partial used 3 epochs, while Full used 5 epochs plus cosine scheduling and warmup.
Therefore the observed performance gap must be attributed to the complete
experimental configurations, not exclusively to full unfreezing.

Status: FROZEN.


## 3. Extractive Question Answering — Russel

Dataset: SQuAD v1.1
Base model: bert-base-cased
Metrics: Exact Match and F1

| Method | Validation EM | Validation F1 | Final Training Loss | Training Time |
|---|---:|---:|---:|---:|
| Partial Fine-Tuning | 76.75% | 85.17% | 0.9040 | 1976.1 s |
| Full Fine-Tuning | 80.75% | 88.21% | 0.4990 | 4075.4 s |

Selected method: Full Fine-Tuning

Reason:
Full improved final Validation EM by +4.00 percentage points and F1 by
+3.04 percentage points, while requiring approximately twice the training time.

Dataset note:
The experiment used the complete SQuAD v1.1 training split (87,599 examples)
rather than the suggested ~15k subsample.

Status: CLOSED.


## 4. Topic Classification — Bianca

Dataset: AG News
Base model: bert-base-uncased
Classes: World, Sports, Business, Sci/Tech

| Method | Accuracy | Macro F1 | Training Time |
|---|---:|---:|---:|
| Feature-based: frozen BERT + Logistic Regression | 90.36% | ~90% | 222.54 s |
| Full Fine-Tuning | 94.45% | 0.9444 | 2793.66 s |

Selected method: Full Fine-Tuning

Learning rates:
- Classification head: 1e-3
- BERT encoder: 2e-5

Training diagnostics:
- Training examples: 108,000
- Validation examples: 12,000
- Official test examples: 7,600
- Training steps: 6,750
- Training loss: 0.180748
- Validation loss: 0.173437
- Official test loss: 0.175384

Reason:
Fine-tuning improved accuracy by approximately +4.09 percentage points and
Macro F1 by approximately +0.04, at a substantially higher computational cost.

Evaluation protocol:
A fixed-seed 90/10 split of the original AG News training data produced
108,000 training examples and 12,000 validation examples. The official
7,600-example test split remained untouched until final evaluation.

Published artifact:
The final model and tokenizer are published and independently verified at
bialexacosta21/bert-agnews-topic-classification.

Status: COMPLETE / HF VERIFIED.


## Global conclusions

- Full Fine-Tuning produced the highest measured predictive performance in all
  four team experiments.
- The computational benefit of lighter adaptation methods was visible,
  particularly in NER and Topic Classification.
- Performance comparisons must consider not only model scores but also training
  time, trainable parameters, memory requirements, and experimental configuration.
- A single seed was used per configuration, so differences close to the expected
  seed variability range should not be interpreted as universal superiority.
