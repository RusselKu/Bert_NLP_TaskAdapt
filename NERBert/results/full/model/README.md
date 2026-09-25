---
language: en
pipeline_tag: token-classification
base_model: google-bert/bert-base-cased
datasets:
- lhoestq/conll2003
tags:
- bert
- ner
- conll2003
metrics:
- f1
- precision
- recall
---
# BERT NER - Jonathan / U2T01

Modelo entrenado con **full fine-tuning** para PER, ORG, LOC y MISC
en noticias en inglés. Cabeza lineal con nueve etiquetas BIO; el tokenizador conserva mayúsculas.

## Datos y entrenamiento
CoNLL-2003, revisión `19edcb426bfd625c275c17b9a99b2239243f4377`. Splits originales:
train 14041, validation 3250,
test 3453. Sin truncamiento de oraciones.
Base `google-bert/bert-base-cased` revisión `cd5ef92a9fb2f889e972770a36d4ed042daf221e`.
Semilla 42, 3 épocas, batch efectivo
16. AdamW: LR head
0.001, encoder 2e-05, warmup 10%, scheduler lineal.
Sólo la primera subpalabra recibe etiqueta; restantes, padding y especiales: -100.
Hardware: NVIDIA GeForce RTX 4060 Laptop GPU. FP16 en CUDA.

## Evaluación
F1 micro de entidades con span y tipo exactos (seqeval, strict IOB2).
Validación F1: 0.951129.
Test F1: 0.915353; precisión:
0.915515; recall: 0.915191;
accuracy por palabra: 0.982836.

Se seleccionó full por su F1 de validación (95.11%). La diferencia observada respecto a partial fue 3.40 puntos porcentuales. La selección se registró antes de evaluar test. Se entrenó una sola semilla por método; no se realizaron pruebas de significancia estadística. Aunque la diferencia supera 3 puntos en esta ejecución, una sola semilla no caracteriza su variabilidad.

## Uso
```python
from transformers import pipeline
ner = pipeline('token-classification', model='RUTA_O_REPO_DEL_MODELO', aggregation_strategy='first')
print(ner('Jonathan visited Microsoft in London.'))
```
Para reproducir las métricas se deben usar las palabras originales del dataset y
evaluar únicamente la primera subpalabra, como en `train_ner.py`.

## Limitaciones y licencia
Uso académico en inglés periodístico. No validado para español, otros dominios o
decisiones sobre personas. Puede omitir entidades o reproducir sesgos del corpus.
El modelo base tiene licencia Apache-2.0; eso no otorga derechos sobre los textos de
Reuters de CoNLL-2003. Consultar los términos del corpus antes de redistribuirlo.
El repositorio del modelo no incluye el corpus.

## Referencias
- https://arxiv.org/abs/1810.04805
- https://huggingface.co/google-bert/bert-base-cased
- https://huggingface.co/datasets/lhoestq/conll2003
- https://aclanthology.org/W03-0419/
- https://huggingface.co/docs/transformers/tasks/token_classification
- https://github.com/chakki-works/seqeval
