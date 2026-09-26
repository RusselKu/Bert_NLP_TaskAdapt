# NER — Adapting BERT to Named Entity Recognition

Adaptación de `bert-base-cased` a Named Entity Recognition sobre **CoNLL-2003**, comparando
dos métodos de adaptación (*partial fine-tuning* vs *full fine-tuning*) con medición empírica,
como parte de la asignación U2T01 (Adapting BERT for NLP tasks).

## Resumen de resultados

| Método  | Parámetros entrenables | F1 validación | F1 test | Precision test | Recall test | Tiempo entrenamiento | VRAM pico |
|---------|------------------------|----------------|---------|-----------------|-------------|-----------------------|-----------|
| Partial | 14.18M                 | 91.71%         | 88.79%  | 89.03%          | 88.54%      | 115.6 s               | 0.76 GiB  |
| **Full**| 107.73M                | **95.11%**     | **91.54%** | 91.55%       | 91.52%      | 358.5 s               | 2.42 GiB  |

**Ganador: full fine-tuning**, seleccionado por F1 de validación (criterio fijado antes de tocar
el set de test). La ganancia sobre partial es de ~2.75 puntos de F1 en test, a costa de ~3.1x el
tiempo de entrenamiento y ~3.2x la VRAM pico.

### F1 por tipo de entidad (modelo ganador, test)

| Entidad | Precision | Recall | F1     | Soporte |
|---------|-----------|--------|--------|---------|
| PER     | 96.99%    | 95.67% | 96.33% | 1617    |
| LOC     | 93.28%    | 93.23% | 93.25% | 1668    |
| ORG     | 89.23%    | 90.31% | 89.77% | 1661    |
| MISC    | 80.65%    | 80.77% | 80.71% | 702     |

`MISC` es consistentemente la categoría más débil: agrupa entidades heterogéneas (nacionalidades,
eventos, adjetivos derivados de nombres propios, etc.) sin un patrón léxico/sintáctico tan
consistente como PER, LOC u ORG, lo que dificulta que el modelo generalice con la misma precisión.

## Dataset

- **Fuente:** [CoNLL-2003](https://huggingface.co/datasets/lhoestq/conll2003) (esquema de
  etiquetado **IOB2**, 4 tipos de entidad: PER, LOC, ORG, MISC).
- **Splits:** 14,041 oraciones de entrenamiento / 3,250 de validación / 3,453 de test.
- Versión y hash exactos del dataset fijados en `results/data_audit.json` y `results/environment.json`
  para reproducibilidad.

## Modelo base

- `bert-base-cased` (110M parámetros, ~107.7M de ellos en el cuerpo del encoder).
- Cased porque la mayúscula inicial es una señal fuerte para reconocer entidades nombradas.

## Manejo de subwords

El tokenizador puede partir una palabra en varias piezas (`Washington` → `Wash` + `##ington`).
La convención seguida:

- La etiqueta real de la palabra va en su **primera sub-palabra**.
- Todo lo demás — continuaciones de subword, `[CLS]`, `[SEP]`, padding — recibe `-100`,
  el índice que `CrossEntropyLoss` ignora automáticamente.

Antes de entrenar, `sanity_check.txt` imprime un batch con tokens alineados junto a sus etiquetas,
para verificar visualmente que el alineamiento es correcto (ver `results/sanity_check.txt`).

## Métodos de adaptación comparados

| Método  | Qué se entrena                          | Learning rate                                  |
|---------|-------------------------------------------|------------------------------------------------|
| Partial | Head + top 2 capas del encoder            | Head: 1e-3 · Encoder (capas superiores): 2e-5   |
| Full    | Head + las 12 capas del encoder completo  | Head: 1e-3 · Encoder: 2e-5                      |

Ambos usan **dos grupos de parámetros** con learning rates distintos: uno alto para la cabeza
recién inicializada, uno bajo para las capas preentrenadas, evitando que pasos grandes destruyan
lo aprendido durante el preentrenamiento de BERT.

- **Épocas:** 3 por método.
- **Seed:** fija (ver `results/resolved_config.json`), para reproducibilidad.
- **Selección del ganador:** por F1 estricto de validación (`seqeval`, esquema IOB2), *sin tocar
  el set de test* hasta después de elegir el método. En caso de empate exacto, gana el método con
  menos parámetros entrenables (ver `results/selection.json`).

## Métricas

- **F1 estricto por entidad** (`seqeval`, modo `strict`, esquema `IOB2`) como métrica principal:
  a diferencia de accuracy por token, no se infla por el fuerte desbalance de la etiqueta `O`
  (la mayoría de los tokens no pertenecen a ninguna entidad).
- Precision y recall por entidad, además del micro/macro/weighted promedio (`test_entity_report.json`).
- Accuracy por token reportada como referencia adicional, no como criterio de selección.
- Métricas de entrenamiento por época (loss, tiempo por paso/época, VRAM pico) para debug y
  comparación de costo computacional (`result.json` de cada método).

## Estructura del proyecto

```
NERBert/
├── ner_bert.ipynb              # Notebook principal (flujo completo)
├── train_ner.py                # Script de entrenamiento y evaluación
├── ner_utils.py                # Tokenización, alineamiento de labels, métricas
├── build_report.py             # Genera la tabla/gráficas comparativas y el PDF de reporte
├── make_notebook.py            # Genera el notebook a partir de las celdas fuente
├── execute_notebook.py         # Ejecuta el notebook de punta a punta
├── test_ner.py                 # Pruebas automáticas (alineamiento, F1, congelamiento de capas)
├── config.json                 # Configuración de los experimentos (métodos, seed, hparams)
├── requirements.txt
├── RESULTADOS.md                # Notas de resultados
├── reporte_ner.pdf              # Reporte generado
├── assets/
│   ├── comparison.png
│   └── learning_curves.png
└── results/
    ├── comparison.csv / comparison.json   # Comparación consolidada de métodos
    ├── selection.json                     # Método ganador y criterio de selección
    ├── environment.json                   # Versiones exactas (Python, torch, CUDA, GPU)
    ├── data_audit.json                    # Auditoría del dataset (splits, longitudes, truncado)
    ├── labels.json                        # Mapeo de etiquetas IOB2
    ├── sanity_check.txt                   # Batch de ejemplo con tokens/labels alineados
    ├── resolved_config.json               # Configuración final usada (seed incluida)
    ├── requirements-lock.txt              # Versiones exactas instaladas
    ├── partial/
    │   ├── result.json                    # Métricas de entrenamiento y validación
    │   ├── test_metrics.json              # Métricas finales en test
    │   ├── test_entity_report.json        # F1/precision/recall por tipo de entidad
    │   ├── error_examples.json            # Ejemplos de errores de predicción
    │   └── model/                         # Config, tokenizer y vocab (sin pesos, ver nota abajo)
    └── full/
        └── ...                            # Misma estructura que partial/
```

> **Nota sobre pesos del modelo:** los checkpoints y archivos `.safetensors` **no** se versionan
> en este repositorio (ver `.gitignore`) por su tamaño. Los pesos del modelo ganador se publican
> en HuggingFace Hub (ver sección siguiente).

## Modelo publicado

El modelo ganador (full fine-tuning) se publica en HuggingFace Hub junto con su tokenizador y
un model card que documenta datos de entrenamiento, métricas, uso previsto y limitaciones.

<!-- TODO: agregar enlace al repo de HuggingFace una vez publicado -->
🔗 `https://huggingface.co/<usuario>/<nombre-del-repo>`

## Cómo reproducir

```basg
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. (Opcional) prueba rápida con una muestra reducida — NO es un resultado de entrega
python train_ner.py --smoke

# 3. Entrenamiento completo (corre todos los métodos definidos en config.json)
python train_ner.py

# 4. Generar la tabla comparativa y el reporte
python build_report.py
```

Los resultados se guardan en `results/<método>/`, y la comparación consolidada en
`results/comparison.json` / `results/comparison.csv`. El script no sobreescribe resultados
existentes (falla si `results/comparison.json` ya existe), como protección ante ejecuciones
accidentales repetidas.

## Referencias

- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) (sección 5.3: comparación
  feature-based vs fine-tuning).
- Tjong Kim Sang, E. F. & De Meulder, F. (2003). *Introduction to the CoNLL-2003 Shared Task:
  Language-Independent Named Entity Recognition*.
- Dataset: [lhoestq/conll2003](https://huggingface.co/datasets/lhoestq/conll2003) en HuggingFace Hub.
- Tunstall, L., von Werra, L. & Wolf, T. *Natural Language Processing with Transformers* (O'Reilly),
  capítulos 1–3.
- [HuggingFace Transformers — Fine-tune a pretrained model](https://huggingface.co/docs/transformers/training)
- Librería de evaluación: [`seqeval`](https://github.com/chakki-works/seqeval) para F1 estricto
  por entidad con esquema IOB2.
