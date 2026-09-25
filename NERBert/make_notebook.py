"""Construye el notebook de trabajo sin duplicar la implementación probada."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.splitlines(keepends=True)})


md("""# U2T01: BERT para Named Entity Recognition
**Responsable: Jonathan**

Objetivo: comparar partial fine-tuning (últimas 2 capas) contra full fine-tuning
con `bert-base-cased` y CoNLL-2003. Cada método parte del mismo modelo preentrenado
y de la misma semilla. Hipótesis: ajustar todo el encoder permite mayor adaptación,
pero exige más cómputo y memoria. La hipótesis se contrasta con mediciones propias.

Este notebook usa los módulos adjuntos `train_ner.py`, `ner_utils.py` y
`build_report.py`. Conservar toda la carpeta NERBert al compartirlo.
Consultar README.md para instalar el entorno; no se instalan paquetes al abrirlo.
""")
code("""from pathlib import Path
import sys, json
ROOT = Path.cwd() if (Path.cwd() / 'train_ner.py').exists() else Path.cwd() / 'NERBert'
assert (ROOT / 'train_ner.py').exists(), 'Abrir desde la raíz del repo o NERBert'
sys.path.insert(0, str(ROOT))
import train_ner
import torch
import pandas as pd
from IPython.display import display, Markdown, Image
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
config = json.loads((ROOT / 'config.json').read_text())
display(config)
""")
md("""## Datos, etiquetas y verificación de alineación
Dataset: `lhoestq/conll2003`, splits originales train/validation/test. Nueve etiquetas
BIO para PER, ORG, LOC y MISC más O. Se conserva la primera subpalabra con su etiqueta;
continuaciones, especiales y padding reciben -100. El collator realiza padding dinámico.
Antes de entrenar se imprime un batch y se verifica que cada palabra tenga una etiqueta.
Se mide la longitud real de todas las oraciones: ninguna puede truncarse silenciosamente.

Las revisiones exactas del dataset y modelo se guardan en `resolved_config.json`.
No se redistribuyen los textos del corpus en el modelo exportado.
""")
code("""# Pruebas de alineación, métricas de entidades y parámetros entrenables.
import unittest
suite = unittest.defaultTestLoader.discover(str(ROOT), pattern='test_ner.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful()
""")
md("""## Entrenamiento de los dos métodos
Tres épocas por método, semilla 42, batch efectivo 16. AdamW con LR de cabeza 1e-3 y
encoder 2e-5, warmup del 10% y scheduler lineal. Bias y LayerNorm no usan decay.
FP16 en CUDA. Se elige el mejor checkpoint por F1 de validación.

Se comparan los dos métodos por F1 de validación **antes de evaluar test**.
Un empate exacto favorece el menor número de parámetros entrenables.
El tiempo por época excluye evaluación/guardado; el total incluye ambos.
Esta celda reutiliza resultados ya terminados para no repetir el entrenamiento.
""")
code("""if not (ROOT / 'results' / 'comparison.json').exists():
    train_ner.main(ROOT / 'config.json')
else:
    print('Resultados existentes: se reutilizan sin entrenar de nuevo.')
print((ROOT / 'results' / 'sanity_check.txt').read_text(encoding='utf-8'))
display(json.loads((ROOT / 'results' / 'data_audit.json').read_text()))
""")
md("""## Métricas y selección
F1 micro de entidades con coincidencia exacta de span y tipo, usando seqeval strict
IOB2. Precisión y recall reflejan falsas entidades y omisiones. Accuracy por palabra
es secundaria, porque O domina el corpus. Se ignoran todas las posiciones -100.

Una sola semilla por configuración no permite inferir significancia. Una diferencia
de 1-3 puntos puede ser ruido, según la advertencia de la tarea.
""")
code("""from build_report import build_report
comparison_table = build_report()
display(comparison_table)
display(Markdown((ROOT / 'RESULTADOS.md').read_text(encoding='utf-8')))
display(Image(filename=str(ROOT / 'assets' / 'learning_curves.png')))
display(Image(filename=str(ROOT / 'assets' / 'comparison.png')))
""")
md("""## Errores y evaluación por tipo de entidad
Se muestran ejemplos con desacuerdos y el desglose por tipo de entidad del modelo
elegido. Los ejemplos son ilustrativos, no una muestra aleatoria de errores.
""")
code("""summary = json.loads((ROOT / 'results' / 'comparison.json').read_text())
winner = summary['winner']
display(pd.DataFrame(json.loads((ROOT / 'results' / winner / 'test_entity_report.json').read_text())).T)
errors = json.loads((ROOT / 'results' / winner / 'error_examples.json').read_text())
from datasets import load_dataset
resolved = json.loads((ROOT / 'results' / 'resolved_config.json').read_text())
test = load_dataset(resolved['dataset_id'], revision=resolved['dataset_revision'], split='test')
for error in errors[:3]:
    print('Ejemplo test', error['test_index'])
    display(pd.DataFrame({'palabra': test[error['test_index']]['tokens'],
                          'real': error['gold'], 'predicción': error['predicted']}))
""")
md("""## Inferencia del modelo guardado
Se recarga desde disco el modelo elegido y su tokenizador, con agregación sobre la
primera subpalabra, coherente con el entrenamiento. Esta demostración no es una métrica.
""")
code("""from transformers import pipeline
model_path = ROOT / 'results' / winner / 'model'
ner = pipeline('token-classification', model=str(model_path), tokenizer=str(model_path),
               aggregation_strategy='first', device=0 if torch.cuda.is_available() else -1)
display(ner('Jonathan visited Microsoft in London.'))
""")
md("""## Entrega y publicación
El reporte PDF está en `reporte_ner.pdf`. Compartir con Damian la carpeta del modelo
seleccionado (`results/<winner>/model`), que contiene pesos, tokenizador y model card,
y los resultados/figuras para el reporte grupal.

Publicar requiere un destino Hugging Face, diferente del repositorio GitHub del equipo.
La celda siguiente está desactivada por defecto; no contiene credenciales.
Si se elige privado, conceder acceso a Dexterg83 en el Hub.
""")
code("""PUBLISH = False
HF_REPO_ID = ''  # usuario/nombre-del-modelo
PRIVATE = True
if PUBLISH:
    assert '/' in HF_REPO_ID, 'Indicar repositorio de Hugging Face'
    from huggingface_hub import HfApi, notebook_login
    notebook_login()
    api = HfApi()
    api.create_repo(HF_REPO_ID, private=PRIVATE, exist_ok=True)
    api.upload_folder(repo_id=HF_REPO_ID, folder_path=str(model_path),
                      commit_message='Modelo NER BERT y evaluación U2T01')
else:
    print('Publicación desactivada; entrega local:', model_path)
""")
md("""## Limitaciones y referencias
Modelo evaluado sólo en noticias en inglés y cuatro tipos de entidades. No validado
en español u otros dominios. Puede reproducir sesgos y omitir entidades. Los pesos
base tienen licencia Apache-2.0; revisar por separado los derechos de CoNLL/Reuters.

- [BERT, Devlin et al.](https://arxiv.org/abs/1810.04805)
- [BERT-base-cased](https://huggingface.co/google-bert/bert-base-cased)
- [Dataset](https://huggingface.co/datasets/lhoestq/conll2003)
- [CoNLL-2003](https://aclanthology.org/W03-0419/)
- [Hugging Face token classification](https://huggingface.co/docs/transformers/tasks/token_classification)
- [seqeval](https://github.com/chakki-works/seqeval)
""")
notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "kernelspec": {"display_name": "Python 3 (BERT NER)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"}}, "cells": cells}
for i, cell in enumerate(cells):
    cell["id"] = f"ner-{i:02d}"
(ROOT / "ner_bert.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
