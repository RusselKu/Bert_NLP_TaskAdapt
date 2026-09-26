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

## 📊 Resultados Empíricos Auditar en Test Set

| Método de Adaptación | Parámetros Entrenables | Épocas | Accuracy | F1-Score (Macro) | Precision (Macro) | Recall (Macro) | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Partial Fine-Tuning** (Capas 10, 11 + 2 Param Groups) | 14.2M (13.17%) | 3 | 95.43% | 0.8909 | 0.8997 | 0.8865 | 0.1520 |
| **Full Fine-Tuning** (5 Épocas + Cosine Scheduler) 🏆 | **107.7M (100%)** | **5** | **97.42%** | **0.9446** | **0.9536** | **0.9384** | **0.1134** |

---

## 🎯 Intended Use (Uso Previsto)

### Aplicaciones Primarias
* **Etiquetado Morfosintáctico Automático**: Clasificación de categorías gramaticales (Part-of-Speech Tagging) en oraciones en idioma inglés.
* **Análisis Lingüístico**: Extracción de características sintácticas para sistemas downstream de NLP, desambiguación léxica y procesamiento del lenguaje.

### Integración en Python (`pipeline`)
```python
from transformers import pipeline

pos_tagger = pipeline(
    "token-classification", 
    model="Rivaldo2309030/bert-base-cased-pos-tagging-ewt",
    aggregation_strategy="first"
)

texto = "BERT models achieve state-of-the-art results on NLP tasks."
predicciones = pos_tagger(texto)

for item in predicciones:
    print(f"{item['word']:<15} -> {item['entity_group']}")
```

---

## ⚠️ Limitations (Limitaciones)

* **Idioma**: Restringido al idioma inglés (`en`). No apto para análisis multilingüe sin un entrenamiento adicional.
* **Longitud de Secuencia**: Truncado a `max_length = 128` subpalabras. Oraciones excesivamente largas serán cortadas.
* **Dominio**: Entrenado sobre el corpus Universal Dependencies `en_ewt` (que combina blogs, noticias, reviews y correos). Puede presentar una leve degradación de precisión en textos con jerga informal extrema o tweets sin puntuación.

---

## ⚙️ Configuración e Hiperparámetros de Entrenamiento

* **Random Seed:** `SEED = 42` (reproducibilidad garantizada)
* **Batch Size:** 32 por dispositivo
* **Optimizador:** AdamW (`weight_decay = 0.01`)
* **Parameter Groups (Método 1 - Partial):**
  * `classifier`: Learning Rate `1e-3`
  * `bert.encoder.layer.10` y `layer.11`: Learning Rate `2e-5`
* **Parameter Groups (Método 2 - Full Ganador):**
  * `classifier`: Learning Rate `1e-3`
  * `bert.encoder`: Learning Rate `2e-5` con planificador `cosine` y `warmup_steps = 100`

---

## 📚 References (Referencias)

1. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018)**. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv preprint arXiv:1810.04805.
2. **Nivre, J., et al. (2020)**. *Universal Dependencies v2: An Open Community Initiative for Cross-Lingual Dependency Annotation*. Computational Linguistics / LREC.
3. **Wolf, T., et al. (2020)**. *Transformers: State-of-the-Art Natural Language Processing*. EMNLP 2020.
