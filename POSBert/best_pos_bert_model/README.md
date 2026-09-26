---
language:
- en
license: apache-2.0
tags:
- token-classification
- pos-tagging
- bert
- universal-dependencies
datasets:
- universal-dependencies/universal_dependencies
metrics:
- accuracy
- f1
library_name: transformers
pipeline_tag: token-classification
widget:
- text: "BERT models achieve state-of-the-art results on NLP tasks."
  example_title: "POS Tagging Example"
---

# 🏷️ BERT Base Cased - Part-of-Speech (POS) Tagging (`en_ewt`)

Modelo `bert-base-cased` ajustado finamente (Fine-tuned) para la tarea de **Part-of-Speech (POS) Tagging** utilizando las 17 etiquetas universales (UPOS) del dataset `universal-dependencies/universal_dependencies` (configuración `en_ewt`).

---

## 📌 Resumen del Modelo

* **Desarrollado por:** Angel Rivaldo
* **Modelo Base:** `bert-base-cased` (110M parámetros)
* **Tarea:** Token Classification (POS Tagging)
* **Idioma:** Inglés (`en`)
* **Dataset:** Universal Dependencies v2.14 (`en_ewt`, 17 UPOS tags)
* **Manejo de Subpalabras:** Regla estricta del `-100` para subpalabras de continuación (`##`), `[CLS]`, `[SEP]` y `padding`.

---

## 📊 Resultados Empíricos (Test Set)

| Método de Adaptación | Parámetros Entrenables | Épocas | Accuracy | F1-Score (Macro) | Precision | Recall | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Partial Fine-Tuning** (Capas 10 y 11) | 14.9M (13.6%) | 3 | 96.97% | 0.9389 | 0.9377 | 0.9331 | 0.1026 |
| **Full Fine-Tuning** (5 Épocas + Cosine) 🏆 | **107.7M (100%)** | **5** | **97.42%** | **0.9446** | **0.9536** | **0.9384** | **0.1134** |

---

## 🚀 Uso Rápido en Python (`pipeline`)

```python
from transformers import pipeline

# Cargar el modelo e inferencia directa desde Hugging Face Hub
pos_tagger = pipeline(
    "token-classification", 
    model="Rivaldo2309030/bert-base-cased-pos-tagging-ewt",
    aggregation_strategy="first"
)

# Ejemplo de prueba
texto = "BERT models achieve state-of-the-art results on NLP tasks."
predicciones = pos_tagger(texto)

for item in predicciones:
    print(f"{item['word']:<15} -> {item['entity_group']}")
```

---

## ⚙️ Configuración del Entrenamiento

* **Optimizer:** AdamW (`weight_decay=0.01`)
* **Dos Grupos de Parámetros (Parameter Groups):**
  * `classifier`: Learning Rate `1e-3`
  * `bert.encoder`: Learning Rate `2e-5`
* **Planificador (Scheduler):** `cosine` con `warmup_steps=100`
* **Épocas:** 5
* **Batch Size:** 32 por dispositivo
* **Hardware Utilizado:** NVIDIA GeForce RTX 4050 Laptop GPU (CUDA)
