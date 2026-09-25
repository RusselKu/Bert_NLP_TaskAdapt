"""Ejecuta y guarda el notebook con el Python activo, sin instalar un kernel global."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
cache = ROOT.parent / ".cache" / "jupyter"
kernel = cache / "kernels" / "bert-ner"
kernel.mkdir(parents=True, exist_ok=True)
(kernel / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Python 3 (BERT NER)", "language": "python"
}), encoding="utf-8")
os.environ["JUPYTER_PATH"] = str(cache)
os.environ["JUPYTER_RUNTIME_DIR"] = str(cache / "runtime")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".cache" / "matplotlib"))

import nbformat
from nbclient import NotebookClient

if not (ROOT / "results" / "comparison.json").exists():
    raise RuntimeError("Ejecutar primero train_ner.py; esta verificación requiere resultados completos")
path = ROOT / "ner_bert.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
client = NotebookClient(notebook, timeout=1800, kernel_name="bert-ner",
                        resources={"metadata": {"path": str(ROOT)}})
try:
    client.execute()
finally:
    # Nombre estándar para abrirlo en otros equipos y elegir su entorno local.
    notebook.metadata.kernelspec = {"name": "python3", "display_name": "Python 3 (BERT NER)", "language": "python"}
    nbformat.write(notebook, path)
print("Notebook ejecutado y guardado:", path)
