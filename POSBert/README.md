# 📄 Entregables y Datos Completos para el Reporte del Equipo (BERT POS Tagging)

> **Autor:** Angel Rivaldo  
> **Integrador / Coordinador del Reporte:** Damián  
> **Tarea:** Part-of-Speech (POS) Tagging  
> **Modelo Base:** `bert-base-cased` (110M parámetros)  
> **Dataset:** Universal Dependencies (`universal-dependencies/universal_dependencies`, config `en_ewt`, 17 etiquetas UPOS)  
> **Ruta del Notebook en GitHub**: [`pos_tagging_bert_v2.ipynb`](https://github.com/RusselKu/Bert_NLP_TaskAdapt/blob/main/pos_tagging_bert_v2.ipynb)

---

## 🔗 1. Enlaces Principales de la Entrega

* **Modelo Entrenado en Hugging Face Hub**:  
  👉 [`https://huggingface.co/Rivaldo2309030/bert-base-cased-pos-tagging-ewt`](https://huggingface.co/Rivaldo2309030/bert-base-cased-pos-tagging-ewt)
* **Repositorio de Código en GitHub**:  
  👉 [`https://github.com/RusselKu/Bert_NLP_TaskAdapt`](https://github.com/RusselKu/Bert_NLP_TaskAdapt)
* **Ruta Exacta del Notebook**:  
  👉 `pos_tagging_bert_v2.ipynb` (en la raíz del repositorio de GitHub)

---

## ⚙️ 2. Hiperparámetros Globales de Entrenamiento

| Parámetro | Valor Configurado |
| :--- | :--- |
| **Random Seed** | `SEED = 42` (fijado en `random`, `numpy`, `torch` y `set_seed`) |
| **Batch Size (Entrenamiento)** | `32` por dispositivo |
| **Batch Size (Evaluación)** | `32` por dispositivo |
| **Longitud Máxima de Secuencia** | `max_length = 128` subpalabras |
| **Optimizador** | AdamW (`weight_decay = 0.01`) |
| **Hardware de Entrenamiento** | NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.4) |

---

## 📈 3. Desglose de Resultados por Época (Epoch-by-Epoch Scores)

### A. Modelo Ganador: Método 2 (Full Fine-Tuning - 5 Épocas + Cosine Scheduler)
* **Parámetros Entrenables**: 107,732,753 (100% de los parámetros)
* **Parameter Groups**: `classifier` con `lr = 1e-3`, `bert.encoder` con `lr = 2e-5`
* **Warmup**: `warmup_steps = 100` | **Scheduler**: `cosine`

| Época | Step | Training Loss (Final de época) | Validation Loss | Accuracy (Eval) | F1-Score Macro | Precision Macro | Recall Macro |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Época 1** | 392 | 0.0969 | 0.1084 | 96.91% | 0.9072 | 0.9633 | 0.9054 |
| **Época 2** | 784 | 0.0494 | 0.0941 | 97.26% | 0.9349 | 0.9491 | 0.9281 |
| **Época 3** | 1176 | 0.0314 | 0.1026 | 97.41% | 0.9438 | 0.9532 | 0.9383 |
| **Época 4** | 1568 | 0.0154 | 0.1097 | 97.39% | 0.9436 | 0.9494 | 0.9392 |
| **Época 5** 🏆 | **1960** | **0.01297** | **0.1134** | **97.42%** | **0.9446** | **0.9536** | **0.9384** |

---

### B. Método 1 Corregido: Partial Fine-Tuning (Top 2 Layers - 2 Parameter Groups - 3 Épocas)
* **Parámetros Entrenables**: 14,188,817 (13.17% de los parámetros)
* **Capas Entrenadas**: `classifier` + `bert.encoder.layer.10` y `bert.encoder.layer.11`
* **Parameter Groups (Exigido por Rúbrica)**:
  * `classifier`: `lr = 1e-3` (Cabeza recién inicializada)
  * `bert.encoder.layer.10` y `layer.11`: `lr = 2e-5` (Capas preentrenadas del encoder)

| Época | Step | Training Loss (Final de época) | Validation Loss | Accuracy (Eval) | F1-Score Macro | Precision Macro | Recall Macro |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Época 1** | 392 | 0.1901 | 0.1459 | 95.48% | 0.8966 | 0.9241 | 0.8908 |
| **Época 2** | 784 | 0.1415 | 0.1482 | 95.39% | 0.8902 | 0.9022 | 0.8851 |
| **Época 3** | **1176** | **0.1336** | **0.1520** | **95.43%** | **0.8909** | **0.8997** | **0.8865** |

---

## ⏱️ 4. Tiempos de Entrenamiento y Rendimiento

| Métrica de Tiempo / Velocidad | Método 1 (Partial Fine-Tuning - 2 Parameter Groups) | Método 2 (Full Fine-Tuning - Modelo Ganador) |
| :--- | :---: | :---: |
| **Tiempo Total de Entrenamiento** | **173.45 segundos** ($\sim 2.89$ minutos) | **123.40 segundos** ($\sim 2.05$ minutos) |
| **Tiempo Promedio por Época** | $\sim 57.81$ segundos / época | $\sim 24.68$ segundos / época |
| **Tiempo Promedio por Step** | $\sim 0.147$ segundos / step | $\sim 0.063$ segundos / step |
| **Velocidad de Evaluación** | $\sim 370$ muestras / segundo | $\sim 400$ muestras / segundo |

---

## 🎯 5. Intended Use & Limitations (Uso Previsto y Limitaciones)

### Intended Use (Uso Previsto)
* **Aplicaciones Primarias**: Etiquetado morfosintáctico automático (Part-of-Speech Tagging) en oraciones en idioma inglés.
* **Uso Académico y Práctico**: Análisis lingüístico, extracción de características sintácticas para sistemas downstream de NLP, resolución de ambigüedad léxica.
* **Pipeline de Inferencia**: Totalmente compatible con Hugging Face `pipeline("token-classification", model="Rivaldo2309030/bert-base-cased-pos-tagging-ewt")`.

### Limitations (Limitaciones)
* **Idioma**: Restringido al idioma inglés (`en`). No apto para análisis multilingüe o lenguas de bajo recurso sin fine-tuning adicional.
* **Longitud Máxima**: Truncado a 128 subpalabras por secuencia. Oraciones extremadamente largas serán cortadas.
* **Dominio**: Ajustado sobre el dataset Universal Dependencies `en_ewt` (que incluye blogs, reviews, noticias y correos). Puede presentar menor precisión en textos con jerga informal extrema o tweets.

---

## 📚 6. References (Referencias)

1. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018)**. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv preprint arXiv:1810.04805.
2. **Nivre, J., et al. (2020)**. *Universal Dependencies v2: An Open Community Initiative for Cross-Lingual Dependency Annotation*. Computational Linguistics / LREC.
3. **Wolf, T., et al. (2020)**. *Transformers: State-of-the-Art Natural Language Processing*. EMNLP 2020.
