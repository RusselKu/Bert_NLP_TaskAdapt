# 📘 Guía de Referencia y Mejores Prácticas: Adaptación de BERT (NLP Task Adaptation)

> **Autores:** Rivaldo & Equipo  
> **Proyecto:** U2T01 - Adapting BERT for NLP Tasks  
> **Tarea Específica:** Part-of-Speech (POS) Tagging sobre el dataset `universal-dependencies/universal_dependencies` (`en_ewt`, 17 etiquetas UPOS).

---

## 📌 1. Resumen de Archivos a Subir

Para mantener el repositorio limpio y cumplir con las entregas del equipo y la rúbrica:

### A. Al Repositorio de GitHub
1. **`pos_tagging_bert_v2.ipynb`**: Notebook reproducible completo con verificaciones de CUDA, Sanity-Check, los 2 métodos de entrenamiento y las 3 gráficas estéticas (`matplotlib`/`seaborn`).
2. **`.gitignore`**: Configuración para excluir binarios pesados (evita sobrepasar el límite de 100 MB de GitHub).
3. **`GUIA_TAREA_BERT.md`**: Este documento de referencia técnica.

### B. Al Hugging Face Hub (HF Hub)
* La carpeta exportada **`./best_pos_bert_model`** que contiene:
  * `model.safetensors` ($\sim 438$ MB, pesos fine-tuneados).
  * `config.json` (configuración del modelo y mapeos `id2label`/`label2id`).
  * `tokenizer_config.json`, `vocab.txt` y `special_tokens_map.json`.

---

## ⚠️ 2. Puntos Críticos y Trampas Evitadas durante el Desarrollo

Para que el resto del equipo no tenga errores ni diferencias en los resultados, deben considerar los siguientes 5 puntos técnicos clave:

### 1️⃣ Regla Estricta de Alineación de Subpalabras (Regla del `-100`)
BERT usa tokenización **WordPiece**, que divide palabras en subpalabras (ejemplo: `"running"` $\rightarrow$ `"run"`, `"##ning"`).
* **Primera subpalabra**: Recibe la etiqueta POS original (ej. `VERB`).
* **Subpalabras de continuación (`##...`)**: Deben recibir estrictamente el ID `-100`.
* **Tokens especiales (`[CLS]`, `[SEP]`) y `padding`**: Deben recibir el ID `-100`.

> **¿Por qué?** PyTorch `CrossEntropyLoss` ignora automáticamente los índices con valor `-100`, evitando sesgar el cálculo de la pérdida en subpalabras y tokens especiales.

### 2️⃣ Corrección en la Función de Métricas (`scikit-learn` vs `seqeval`)
❌ **No usar `seqeval` para POS Tagging**: La librería `seqeval` está diseñada exclusivamente para Named Entity Recognition (NER) con formato IOB/BIO (`B-PER`, `I-PER`). Si se usa con etiquetas POS crudas (`NOUN`, `VERB`), ignora la mayoría de las etiquetas o lanza errores de parseo.  
✅ **Usar `scikit-learn`**:
```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Filtrar únicamente posiciones donde label != -100
    flat_preds = [
        pred_item for pred_seq, label_seq in zip(predictions, labels)
        for pred_item, label_item in zip(pred_seq, label_seq) if label_item != -100
    ]
    flat_labels = [
        label_item for label_seq in labels
        for label_item in label_seq if label_item != -100
    ]

    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_labels, flat_preds, average='macro', zero_division=0
    )
    acc = accuracy_score(flat_labels, flat_preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
```

### 3️⃣ Métodos de Adaptación Comparados
La rúbrica exige comparar 2 métodos de fine-tuning:

* **Método 1: Partial Fine-Tuning (Top 2 Encoder Layers)**
  * Se congelan los embeddings y las capas 0 a 9 del encoder de BERT.
  * Se descongelan únicamente la cabeza de clasificación (`classifier`) y las **capas 10 y 11** ($\sim 14.9$M de parámetros entrenables, 13.6% del total).
  * **Resultados en Test:** Accuracy **96.97%**, F1 Macro **0.9389**.

* **Método 2: Full Fine-Tuning con Grupos de Parámetros y Cosine Scheduler**
  * Se entrena el 100% de los parámetros (107.7M).
  * Se configuran **dos grupos de tasas de aprendizaje (Parameter Groups)**:
    * `lr = 1e-3` para la cabeza de clasificación (`classifier`).
    * `lr = 2e-5` para las capas preentrenadas de BERT.
  * Se utiliza un planificador **`lr_scheduler_type="cosine"`** con `warmup_steps=100` a lo largo de 5 épocas.
  * **Resultados en Test:** Accuracy **97.42%**, F1 Macro **0.9446** (🏆 **Modelo Ganador**).

### 4️⃣ Cambios de API en Hugging Face `transformers` (v5+)
* En `Trainer(...)`: Usar `processing_class=tokenizer` en lugar de `tokenizer=tokenizer`.
* En `TrainingArguments(...)`: Usar `warmup_steps=100` en lugar de `warmup_ratio=0.1`.

---

## 🛠️ 3. Pasos para Subir el Código a GitHub

Ejecutar los siguientes comandos en la terminal local dentro de la carpeta del proyecto:

```bash
# 1. Verificar el estado del repositorio
git status

# 2. Agregar los archivos de código y guía
git add pos_tagging_bert_v2.ipynb .gitignore GUIA_TAREA_BERT.md

# 3. Guardar el commit con un mensaje descriptivo
git commit -m "feat: agregar notebook V2 con POS tagging, metricas scikit-learn y guia de equipo"

# 4. Subir los cambios a la rama principal de GitHub
git push origin main
```

---

## 🤗 4. Pasos para Subir el Modelo a Hugging Face Hub

Para compartir el enlace público del modelo entrenado en la entrega del equipo:

### Opción A: Desde Python (Recomendada)
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Cargar el modelo guardado localmente
model_path = "./best_pos_bert_model"
model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Subir directamente a tu cuenta de Hugging Face Hub
# (Sustituye 'tu-usuario' por tu nombre de usuario de Hugging Face)
repo_name = "tu-usuario/bert-base-cased-pos-tagging-ewt"

model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print(f"✅ Modelo publicado exitosamente en: https://huggingface.co/{repo_name}")
```

### Opción B: Mediante la CLI de Hugging Face
```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload tu-usuario/bert-base-cased-pos-tagging-ewt ./best_pos_bert_model .
```

---

## 📊 5. Resumen de Resultados para el Reporte del Equipo

Pueden tomar la siguiente tabla comparativa para la sección de resultados del reporte grupal:

| Método | Parámetros Entrenables | Épocas | Test Accuracy | Test F1 (Macro) | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Partial Fine-Tuning (Top 2 Layers)** | 14,972,945 (13.6%) | 3 | 96.97% | 0.9389 | 0.1026 |
| **2. Full Fine-Tuning (Cosine + Warmup)** | 107,732,753 (100%) | 5 | **97.42%** | **0.9446** | 0.1134 |
