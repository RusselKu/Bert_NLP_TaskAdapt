# U2T01: Adapting BERT for NLP Tasks

**Universidad Politécnica de Yucatán**
**Data Engineering**
**Course:** Trends in Data Science
**Group:** B
**Instructor:** Dexter Gomez

## Project Overview

This project evaluates different strategies for adapting BERT to four
Natural Language Processing tasks with different output structures:

1. Named Entity Recognition (NER)
2. Part-of-Speech Tagging (POS)
3. Extractive Question Answering (QA)
4. Topic Classification

The experiments compare lighter adaptation strategies against full
fine-tuning and analyze the trade-off between predictive performance,
trainable parameters, training time, and computational cost.

The project uses BERT-base models as the pretrained encoder and follows
the Hugging Face ecosystem for tokenization, datasets, training, model
serialization, and Hub publication.

---

## Assignment Coverage

| Requirement | Project implementation |
|---|---|
| BERT-base as pretrained body | Used in all four delivered task pipelines |
| At least two adaptation methods per task | Implemented for NER, POS, QA, and Topic Classification |
| Separate learning rates for new heads and pretrained encoder parameters | Implemented with separate optimizer parameter groups in fine-tuning experiments |
| Task-appropriate evaluation | Entity F1 for NER, Macro F1/Accuracy for POS, EM/F1 for QA, Accuracy/Macro F1 for Topic |
| Training diagnostics | Loss, epoch scores, execution time, trainable parameters, and additional resource metrics where available |
| Fixed seeds and reproducibility documentation | Experiment configurations use fixed seeds and documented training settings |
| Error analysis | Included in the final report using only preserved experimental evidence |
| Cross-task comparison | Included in the final technical report |
| Final PDF report | Available in `docs/final_report.pdf` |
| Hugging Face publication | NER, POS, and QA independently verified; Topic artifact pending |
| Model Cards | Verified for the three currently published repositories |

---

## Tasks and Datasets

| Task | Dataset | Base Model | Compared Methods | Primary Evaluation |
|---|---|---|---|---|
| Named Entity Recognition | CoNLL-2003 (`lhoestq/conll2003`) | `bert-base-cased` | Partial Fine-Tuning vs Full Fine-Tuning | Strict entity-level F1 |
| Part-of-Speech Tagging | UD English EWT | `bert-base-cased` | Partial Fine-Tuning vs Full Fine-Tuning | Macro F1 and Accuracy |
| Extractive Question Answering | SQuAD v1.1 | `bert-base-cased` | Partial Fine-Tuning vs Full Fine-Tuning | Exact Match and F1 |
| Topic Classification | AG News (`fancyzhx/ag_news`) | `bert-base-uncased` | Frozen BERT features + Logistic Regression vs Full Fine-Tuning | Accuracy and Macro F1 |

For the token-classification tasks, subword alignment follows the convention
of assigning the original word label to the first subword and using `-100`
for special tokens and continuation subwords so that ignored positions do
not contribute to the token-classification loss.

---

## Optimization Strategy

Fine-tuning experiments use separate optimizer parameter groups so that the
newly initialized task head can learn faster than the pretrained BERT
encoder.

| Task | Head Learning Rate | Encoder Learning Rate |
|---|---:|---:|
| NER | `1e-3` | `2e-5` |
| POS | `1e-3` | `2e-5` |
| QA | `3e-4` | `3e-5` |
| Topic Classification | `1e-3` | `2e-5` |

The Topic feature-based experiment does not update BERT parameters. Instead,
the frozen encoder is used to generate representations that are consumed by
a Logistic Regression classifier.

---

## Experimental Results

### Named Entity Recognition

The selected model was Full Fine-Tuning, chosen using validation strict
entity-level F1 before final test evaluation.

| Method | Validation F1 | Test F1 | Test Precision | Test Recall | Training Time |
|---|---:|---:|---:|---:|---:|
| Partial Fine-Tuning | 91.71% | 88.79% | 89.03% | 88.54% | 115.63 s |
| **Full Fine-Tuning** | **95.11%** | **91.54%** | **91.55%** | **91.52%** | 358.54 s |

The Full model improved test F1 by approximately 2.75 percentage points,
while requiring substantially more training time and GPU memory.

Detailed implementation and results:

- [`NERBert/README.md`](NERBert/README.md)
- [`NERBert/ner_bert.ipynb`](NERBert/ner_bert.ipynb)

---

### Part-of-Speech Tagging

The selected model was Full Fine-Tuning.

| Method | Test Accuracy | Test Macro F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Partial Fine-Tuning | 95.43% | 89.09% | 89.97% | 88.65% |
| **Full Fine-Tuning** | **97.42%** | **94.46%** | **95.36%** | **93.84%** |

The Partial and Full configurations differ in epoch count and scheduling
strategy. Therefore, their performance and timing differences should be
interpreted as comparisons between the complete experimental
configurations, not as an isolated causal effect of unfreezing additional
BERT layers.

Detailed implementation:

- [`POSBert/README.md`](POSBert/README.md)
- [`POSBert/pos_tagging_bert_v2.ipynb`](POSBert/pos_tagging_bert_v2.ipynb)

---

### Extractive Question Answering

The QA experiments use SQuAD v1.1 and compare Partial Fine-Tuning of the top
four encoder layers against Full Fine-Tuning.

| Method | Validation Exact Match | Validation F1 | Final Training Loss | Training Time |
|---|---:|---:|---:|---:|
| Partial Fine-Tuning | 76.75% | 85.17% | 0.9040 | 1976.1 s |
| **Full Fine-Tuning** | **80.75%** | **88.21%** | **0.4990** | 4075.4 s |

The final Full Fine-Tuning configuration was selected using the recorded
validation performance.

Detailed implementation:

- [`QABert/README.md`](QABert/README.md)
- [`QABert/extractive_qa_bert_v2.ipynb`](QABert/extractive_qa_bert_v2.ipynb)

---

### Topic Classification

Topic Classification compares a feature-based approach against BERT
fine-tuning on AG News.

| Method | Accuracy | Macro F1 | Training Time |
|---|---:|---:|---:|
| Frozen BERT + Logistic Regression | 90.36% | 0.90 | 222.54 s |
| **Full Fine-Tuning** | **94.58%** | **0.95** | 2889.20 s |

The fine-tuned model achieved the highest measured classification
performance.

**Evaluation limitation:** the AG News official test split was also supplied
to the Trainer as the evaluation dataset during the fine-tuning experiment.
Although only one epoch was used and no early-stopping model selection was
performed, the test set was not maintained as a completely untouched final
evaluation set. This limitation is explicitly documented in the final
report.

Notebook:

- [`U2T01_Topic_Classification.ipynb`](U2T01_Topic_Classification.ipynb)

---

## Training Diagnostics

Training diagnostics were collected to verify optimization behavior and
compare adaptation cost.

Examples include:

- training and evaluation loss;
- task score by epoch;
- total and per-epoch execution time;
- trainable parameter counts;
- optimizer-step throughput;
- GPU memory usage where available.

The consolidated numerical record is available in:

- [`docs/final_report_data.md`](docs/final_report_data.md)

Loss values are not treated as perfectly standardized cross-task quantities
because the four experiments use different logging implementations and task
objectives.

---

## Hugging Face Models

The following final repositories have been independently verified using
`scripts/verify_hub_model.py`.

| Task | Hugging Face Repository | Model Class | Status |
|---|---|---|---|
| NER | [jonav/bert-base-cased-ner-conll2003](https://huggingface.co/jonav/bert-base-cased-ner-conll2003) | `BertForTokenClassification` | Verified |
| POS | [Rivaldo2309030/bert-base-cased-pos-tagging-ewt](https://huggingface.co/Rivaldo2309030/bert-base-cased-pos-tagging-ewt) | `BertForTokenClassification` | Verified |
| QA | [RusselKuAguilar/bert-base-cased-squad-extractive-qa](https://huggingface.co/RusselKuAguilar/bert-base-cased-squad-extractive-qa) | `BertForQuestionAnswering` | Verified |
| Topic Classification | Pending final artifact publication | `BertForSequenceClassification` | Pending |

For the three verified repositories, the automated verification confirms:

- repository accessibility;
- Model Card presence;
- model weights;
- tokenizer artifacts;
- BERT configuration;
- correct task architecture;
- tokenizer loading;
- model-weight loading.

Verification examples:

```bash
python scripts/verify_hub_model.py \
  --repo-id jonav/bert-base-cased-ner-conll2003

python scripts/verify_hub_model.py \
  --repo-id Rivaldo2309030/bert-base-cased-pos-tagging-ewt

python scripts/verify_hub_model.py \
  --repo-id RusselKuAguilar/bert-base-cased-squad-extractive-qa
```

The Topic Classification model was saved as `bert_agnews_final` during the
original training session, but the trained artifact is not currently present
in the shared repository. Its Hugging Face publication therefore remains the
only outstanding external model-delivery dependency.

---

## Final Technical Report

The complete technical report documents:

- experimental methodology;
- technical decisions;
- metric selection and rationale;
- adaptation-method comparisons;
- training diagnostics;
- computational trade-offs;
- error analysis;
- cross-task comparison;
- limitations;
- conclusions;
- Hugging Face artifacts;
- references.

Available formats:

- **PDF:** [`docs/final_report.pdf`](docs/final_report.pdf)
- **Editable DOCX:** [`docs/final_report.docx`](docs/final_report.docx)
- **Markdown source:** [`docs/final_report.md`](docs/final_report.md)
- **Consolidated experimental data:** [`docs/final_report_data.md`](docs/final_report_data.md)

---

## Repository Structure

```text
Bert_NLP_TaskAdapt/
├── NERBert/
│   ├── ner_bert.ipynb
│   ├── train_ner.py
│   ├── results/
│   └── assets/
├── POSBert/
│   ├── pos_tagging_bert_v2.ipynb
│   ├── best_pos_bert_model/
│   └── assets/
├── QABert/
│   ├── extractive_qa_bert_v2.ipynb
│   ├── best_qa_bert_model/
│   └── assets/
├── U2T01_Topic_Classification.ipynb
├── notebooks/
│   └── deployment/
│       └── huggingface_publish.ipynb
├── scripts/
│   ├── create_test_model.py
│   ├── publish_model.py
│   ├── verify_hub_model.py
│   └── build_final_report_docx.py
├── deployment/
│   ├── README.md
│   └── team_handoff_checklist.md
├── templates/
│   └── model_card_template.md
├── docs/
│   ├── final_report.pdf
│   ├── final_report.docx
│   ├── final_report.md
│   ├── final_report_data.md
│   └── assets/
├── requirements-deployment.txt
└── README.md
```

---

## Reproducibility

Experiments use fixed random seeds where applicable, and the training
configurations are documented in the corresponding notebooks and
task-specific files.

Install the deployment and verification dependencies with:

```bash
pip install -r requirements-deployment.txt
```

NER also provides a task-specific dependency file:

```bash
pip install -r N5RBert/requirements.txt
```

The notebooks contain the preprocessing, tokenizer configuration, model
initialization, optimizer parameter groups, training procedure, evaluation
procedure, and recorded metrics required to reproduce the experiments.

A single recorded run was retained for each experimental configuration.
Consequently, small differences should be interpreted in the context of
expected random-seed variability rather than as universal differences
between adaptation strategies.

---

## Error Analysis and Experimental Limitations

The final report separates measured evidence from interpretation and does not
attribute causes to error patterns that were not preserved in the experiment
outputs.

The main documented limitations are:

- one recorded random seed per experimental configuration;
- expected seed-to-seed variation in neural fine-tuning;
- non-identical POS Partial and Full Fine-Tuning configurations;
- no preserved numerical confusion-matrix cells for POS;
- reuse of the AG News test split as the evaluation dataset during Topic
  Classification fine-tuning;
- pending Hugging Face publication of the Topic Classification model.

These limitations are included to preserve the validity and reproducibility
of the reported conclusions.

---

## Team

| Member | Primary Responsibility |
|---|---|
| Damian Nicolas Sanchez Novelo | Orchestration, testing, benchmarks, Hugging Face integration, consolidated metrics, and final technical report |
| Russel Emmanuel Ku Aguilar | Extractive Question Answering |
| Angel Rivaldo Canche Chuc | Part-of-Speech Tagging |
| Bianca Alexandra Acosta Castellanos | Topic Classification |
| Jonathan Abisai Velasco Martin | Named Entity Recognition |

---

## References

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).
  *BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding.* https://arxiv.org/abs/1810.04805

- Tjong Kim Sang, E. F., & De Meulder, F. (2003).
  *Introduction to the CoNLL-2003 Shared Task: Language-Independent Named
  Entity Recognition.*

- Universal Dependencies.
  https://universaldependencies.org/

- Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016).
  *SQuAD: 100,000+ Questions for Machine Comprehension of Text.*

- AG News dataset:
  https://huggingface.co/datasets/fancyzhx/ag_news

- Hugging Face Transformers documentation:
  https://huggingface.co/docs/transformers/

---

## Current Delivery Status

The experimental code, consolidated metrics, technical documentation, final
PDF report, and three independently verified Hugging Face models are
integrated into the repository.

The only remaining external delivery dependency is publication and
independent verification of the trained Topic Classification model and
tokenizer on Hugging Face Hub.
