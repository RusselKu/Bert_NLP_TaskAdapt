# NER con BERT - Jonathan

Reconocimiento de personas, organizaciones, lugares y entidades misceláneas en inglés.
Se comparan **partial fine-tuning (últimas dos capas)** y **full fine-tuning** sobre
`google-bert/bert-base-cased`, con CoNLL-2003 (`lhoestq/conll2003`).

## Ejecución

Desde la raíz del repositorio, con Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r NERBert/requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name bert-ner --display-name "BERT NER"
```

Abrir `ner_bert.ipynb` y elegir ese kernel. Todo el desarrollo se explica y ejecuta
desde el notebook; las funciones se mantienen en módulos Python para poder probarlas.
Alternativamente:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s NERBert -p test_ner.py -v
.\.venv\Scripts\python.exe NERBert/train_ner.py --smoke
.\.venv\Scripts\python.exe NERBert/train_ner.py
```

`--smoke` verifica el flujo con 128 oraciones por split y una época. Sus resultados
están separados y **no cuentan como los experimentos finales**.
El entrenamiento final usa todos los splits originales y tres épocas por método.
No vuelve a ejecutar una comparación terminada para evitar sobrescribir evidencia.
Para repetir, usar otra copia del proyecto y `--config` apuntando a
`results/resolved_config.json`, que fija los commits del modelo y dataset.
`results/requirements-lock.txt` registra las versiones efectivamente usadas.
Para reproducir el entorno exacto en una instalación nueva:

```powershell
python -m pip install -r NERBert/results/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

## Decisiones experimentales

- Cabeza lineal por token; conserva mayúsculas, que aportan información para NER.
- Primera subpalabra con etiqueta, continuaciones/especiales/padding con `-100`.
- `sanity_check.txt` muestra un batch real. Se verifica que ninguna oración se trunque.
- Semilla 42, igual inicialización, splits, épocas, batch efectivo 16 y scheduler.
- AdamW: cabeza `1e-3`, encoder `2e-5`; bias y LayerNorm sin weight decay.
- Mejor checkpoint por F1 de entidades de validación. La decisión entre métodos
  se guarda **antes** de evaluar ambos modelos sobre test; empate exacto favorece
  el menor número de parámetros entrenables.
- F1 micro estricto IOB2 (span y tipo exactos) es la métrica principal. Precisión y
  recall explican falsos positivos y omisiones. Accuracy por palabra es secundaria,
  porque la abundancia de `O` puede ocultar errores de entidades.
- Una ejecución por configuración. Una diferencia de 1-3 puntos no demuestra una
  superioridad robusta; se reporta como incertidumbre, sin afirmar significancia.
- Los tiempos por época excluyen evaluación/checkpoints; tiempo total de entrenamiento
  incluye esas operaciones. Se registra memoria GPU asignada máxima por PyTorch.

La hipótesis es que full fine-tuning mejora la adaptación de entidades, mientras
partial reduce memoria y cómputo. Los resultados medidos deben confirmar o rechazar
esa hipótesis; no se presuponen puntuaciones.

## Entrega al equipo

`results/` contendrá comparación, auditoría de datos, predicciones, errores, historial,
tiempos y configuración resuelta. Cada método conserva su modelo y tokenizador en
`results/<método>/model/`. La selección final está en `results/selection.json`.
El reporte y las figuras se generan desde los resultados reales con `build_report.py`.
El notebook reutiliza resultados terminados cuando existen para no reentrenar.

Los pesos y cachés quedan fuera de Git. La publicación en Hugging Face requiere
un repositorio de destino y autenticación; el enlace de GitHub no es un destino Hub.
Damian puede publicar la carpeta del modelo ganador junto con su model card.
Si el repositorio Hub es privado, la tarea requiere acceso para `Dexterg83`.

## Límites y fuentes

Uso académico en noticias en inglés; no se ha validado para español, otros dominios,
datos sensibles ni decisiones sobre personas. Cuatro tipos de entidades no cubren
todas las necesidades de extracción. Persisten sesgos del corpus y del preentrenamiento.
El repositorio del dataset no especifica una licencia en sus metadatos.
La licencia Apache-2.0 del modelo base no concede derechos sobre los textos de Reuters;
consultar los términos de CoNLL-2003 antes de redistribuir el corpus. Este proyecto
no publica el corpus ni incluye textos del dataset en el repositorio del modelo.

- BERT: https://arxiv.org/abs/1810.04805
- Modelo: https://huggingface.co/google-bert/bert-base-cased
- Dataset: https://huggingface.co/datasets/lhoestq/conll2003
- CoNLL-2003: https://aclanthology.org/W03-0419/
- Token classification: https://huggingface.co/docs/transformers/tasks/token_classification
- Trainer: https://huggingface.co/docs/transformers/main_classes/trainer
- seqeval: https://github.com/chakki-works/seqeval
