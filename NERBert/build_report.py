"""Genera tabla, figuras, model card y reporte PDF únicamente con resultados reales."""
import json
from pathlib import Path
import shutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parent


def build_report():
    results = ROOT / "results"
    comparison = json.loads((results / "comparison.json").read_text(encoding="utf-8"))
    if comparison.get("smoke"):
        raise ValueError("Una prueba corta no puede convertirse en el reporte final")
    config = json.loads((results / "resolved_config.json").read_text(encoding="utf-8"))
    audit = json.loads((results / "data_audit.json").read_text(encoding="utf-8"))
    environment = json.loads((results / "environment.json").read_text(encoding="utf-8"))
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    rows = []
    for run in comparison["runs"]:
        rows.append({"method": run["method"], "trainable_parameters": run["parameters"]["trainable"],
                     "validation_f1": run["validation_metrics"]["validation_f1"],
                     **{f"test_{k}": run["test_metrics"][k] for k in ("f1", "precision", "recall", "accuracy")},
                     "training_seconds": run["training_wall_seconds"],
                     "peak_vram_gib": run["peak_allocated_vram_bytes"] / 1024**3})
    table = pd.DataFrame(rows)
    table.to_csv(results / "comparison.csv", index=False)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for run in comparison["runs"]:
        train = [x for x in run["history"] if "loss" in x]
        evaluation = [x for x in run["history"] if "eval_f1" in x]
        axes[0].plot([x["epoch"] for x in train], [x["loss"] for x in train], label=run["method"])
        axes[1].plot([x["epoch"] for x in evaluation], [100*x["eval_f1"] for x in evaluation], marker="o", label=run["method"])
        axes[2].plot([x["epoch"] for x in evaluation], [x["eval_loss"] for x in evaluation], marker="o", label=run["method"])
    for ax, title in zip(axes, ["Pérdida de entrenamiento", "F1 entidades: validación (%)", "Pérdida de validación"]):
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.legend()
    fig.tight_layout()
    fig.savefig(assets / "learning_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].bar(table.method, table.training_seconds / 60, color=["#3291a8", "#304b80"])
    axes[0].set_ylabel("Minutos de entrenamiento total")
    axes[1].bar(table.method, table.test_f1 * 100, color=["#3291a8", "#304b80"])
    axes[1].set_ylabel("F1 de entidades en test (%)")
    axes[1].set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(assets / "comparison.png", dpi=180)
    plt.close(fig)

    winner = next(r for r in comparison["runs"] if r["method"] == comparison["winner"])
    rejected = next(r for r in comparison["runs"] if r["method"] != comparison["winner"])
    gap = 100 * (winner["validation_metrics"]["validation_f1"] - rejected["validation_metrics"]["validation_f1"])
    conclusion = (f"Se seleccionó {comparison['winner']} por su F1 de validación "
                  f"({100*winner['validation_metrics']['validation_f1']:.2f}%). "
                  f"La diferencia observada respecto a {rejected['method']} fue {gap:.2f} puntos porcentuales. "
                  "La selección se registró antes de evaluar test. Se entrenó una sola semilla por método; "
                  "no se realizaron pruebas de significancia estadística. "
                  + ("La diferencia queda dentro del rango orientativo de 1-3 puntos de variabilidad indicado en la tarea, "
                     "por lo que no demuestra una superioridad robusta." if gap <= 3 else
                     "Aunque la diferencia supera 3 puntos en esta ejecución, una sola semilla no caracteriza su variabilidad."))
    model_folder = results / comparison["winner"] / "model"
    card = f"""---
language: en
pipeline_tag: token-classification
base_model: {config['model_id']}
datasets:
- {config['dataset_id']}
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

Modelo entrenado con **{comparison['winner']} fine-tuning** para PER, ORG, LOC y MISC
en noticias en inglés. Cabeza lineal con nueve etiquetas BIO; el tokenizador conserva mayúsculas.

## Datos y entrenamiento
CoNLL-2003, revisión `{config['dataset_revision']}`. Splits originales:
train {audit['train']['sentences']}, validation {audit['validation']['sentences']},
test {audit['test']['sentences']}. Sin truncamiento de oraciones.
Base `{config['model_id']}` revisión `{config['model_revision']}`.
Semilla {config['seed']}, {config['epochs']} épocas, batch efectivo
{config['batch_size']*config['gradient_accumulation_steps']}. AdamW: LR head
{config['head_lr']}, encoder {config['encoder_lr']}, warmup 10%, scheduler lineal.
Sólo la primera subpalabra recibe etiqueta; restantes, padding y especiales: -100.
Hardware: {environment['gpu']}. FP16 en CUDA.

## Evaluación
F1 micro de entidades con span y tipo exactos (seqeval, strict IOB2).
Validación F1: {winner['validation_metrics']['validation_f1']:.6f}.
Test F1: {winner['test_metrics']['f1']:.6f}; precisión:
{winner['test_metrics']['precision']:.6f}; recall: {winner['test_metrics']['recall']:.6f};
accuracy por palabra: {winner['test_metrics']['accuracy']:.6f}.

{conclusion}

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
- https://huggingface.co/{config['model_id']}
- https://huggingface.co/datasets/{config['dataset_id']}
- https://aclanthology.org/W03-0419/
- https://huggingface.co/docs/transformers/tasks/token_classification
- https://github.com/chakki-works/seqeval
"""
    (model_folder / "README.md").write_text(card, encoding="utf-8")
    (ROOT / "RESULTADOS.md").write_text("# Resultados NER - Jonathan\n\n" + conclusion + "\n\n```csv\n" +
        table.to_csv(index=False) + "```\n\nVer results/ para el historial y la configuración exacta.\n", encoding="utf-8")
    # Copia pequeña de configuración y métricas junto al modelo para publicación.
    for filename in ["resolved_config.json", "environment.json", "comparison.csv"]:
        shutil.copy2(results / filename, model_folder / filename)

    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor("#17395a")
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 14
    story = []

    def p(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 0.18*cm))

    def tab(data, widths):
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17395a")),
                              ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                              ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                              ("FONTSIZE", (0,0), (-1,-1), 9),
                              ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                              ("TOPPADDING", (0,0), (-1,-1), 8),
                              ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
                              ("LINEBELOW", (0,-1), (-1,-1), 0.5, colors.lightgrey)]))
        story.append(t)
        story.append(Spacer(1, 0.3*cm))

    p("Adaptación de BERT para NER", "Title")
    p("U2T01 | Jonathan | Resultados experimentales", "Heading2")
    p("1. Objetivo y diseño", "Heading2")
    p("Reconocer personas (PER), organizaciones (ORG), lugares (LOC) y entidades misceláneas (MISC). "
      "Se compararon ajuste parcial de las dos últimas capas y ajuste completo de BERT-base-cased. "
      "La hipótesis fue que el ajuste completo mejoraría F1, a cambio de mayor costo de entrenamiento.")
    p(f"Se usaron {audit['train']['sentences']:,} oraciones para entrenamiento, "
      f"{audit['validation']['sentences']:,} para validación y {audit['test']['sentences']:,} para test. "
      "Se conservaron las particiones originales de CoNLL-2003; test no intervino en la selección.")
    p("2. Preprocesamiento y reproducibilidad", "Heading2")
    p("WordPiece con mayúsculas conservadas. Etiqueta sólo en la primera subpalabra; -100 en "
      "continuaciones, tokens especiales y padding. El sanity-check imprime un batch real y verifica "
      "una etiqueta por palabra. No se truncaron oraciones; se verificó el límite de 512 posiciones.")
    p(f"Ambos métodos: semilla {config['seed']}, {config['epochs']} épocas, batch efectivo "
      f"{config['batch_size']*config['gradient_accumulation_steps']}, AdamW, LR cabeza 0.001 y encoder 0.00002, "
      "warmup del 10%, scheduler lineal y gradient clipping 1.0. Se excluyó decay de bias y LayerNorm. "
      f"Hardware: {environment['gpu']}; PyTorch {environment['torch']}, Python {environment['python']}.")
    p("Los SHA del modelo/dataset, paquetes, configuración, conteos de parámetros, tiempos e historial "
      "se entregan en results/. El checkpoint de cada método se eligió por F1 de validación.")
    p("3. Métricas", "Heading2")
    p("La métrica principal es F1 micro de entidades (seqeval estricto IOB2): una predicción es correcta "
      "sólo si coinciden span completo y tipo. Precisión mide falsos positivos y recall mide omisiones. "
      "Accuracy por palabra es secundaria por la abundancia de la etiqueta O. Las pérdidas por época "
      "permiten examinar convergencia. Los tiempos por época excluyen evaluación y guardado; el total los incluye.")
    story.append(PageBreak())
    p("4. Resultados y costo", "Heading1")
    tab([["Método", "Parám. entrenables", "Val F1 %", "Test F1 %", "Minutos"]] +
        [[r["method"], f"{r['trainable_parameters']:,}", f"{100*r['validation_f1']:.2f}",
          f"{100*r['test_f1']:.2f}", f"{r['training_seconds']/60:.2f}"] for r in rows],
        [2.4*cm, 4.1*cm, 3*cm, 3*cm, 3*cm])
    tab([["Método", "Test precisión %", "Test recall %", "Accuracy %", "VRAM GiB"]] +
        [[r["method"], f"{100*r['test_precision']:.2f}", f"{100*r['test_recall']:.2f}",
          f"{100*r['test_accuracy']:.2f}", f"{r['peak_vram_gib']:.2f}"] for r in rows],
        [2.4*cm, 4.1*cm, 3*cm, 3*cm, 3*cm])
    story.append(Image(str(assets / "learning_curves.png"), width=16.5*cm, height=4.7*cm))
    p("Figura 1. Pérdidas y F1 de validación a lo largo del entrenamiento.")
    story.append(Image(str(assets / "comparison.png"), width=14.5*cm, height=5.64*cm))
    p("Figura 2. Costo total y F1 de test; test se consultó después de seleccionar el método.")
    p("5. Selección y alcance de la evidencia", "Heading2")
    p(conclusion)
    story.append(PageBreak())
    p("6. Desglose del modelo elegido", "Heading1")
    report = json.loads((results / comparison["winner"] / "test_entity_report.json").read_text(encoding="utf-8"))
    tab([["Entidad", "Precisión %", "Recall %", "F1 %", "Soporte"]] +
        [[name, f"{100*report[name]['precision']:.2f}", f"{100*report[name]['recall']:.2f}",
          f"{100*report[name]['f1-score']:.2f}", str(report[name]['support'])]
         for name in ["PER", "ORG", "LOC", "MISC"] if name in report],
        [2.4*cm, 4.1*cm, 3*cm, 3*cm, 3*cm])
    p("Las predicciones y etiquetas completas permiten recalcular las métricas. Los primeros veinte "
      "casos con desacuerdos se conservan en error_examples.json para inspección, referidos por índice "
      "del split test. No se han atribuido causas lingüísticas a errores sin inspeccionarlos.")
    p("7. Uso, limitaciones y entrega", "Heading2")
    p("El modelo ganador y su tokenizador están en results/" + comparison["winner"] + "/model/. "
      "La model card documenta métricas y entrenamiento. La publicación Hub requiere el repositorio "
      "del equipo; si es privado se debe conceder acceso a Dexterg83. Este reporte no afirma que ya esté publicado.")
    p("El alcance es inglés periodístico con cuatro tipos de entidad. No se validó generalización a "
      "otros idiomas o dominios. Los resultados pueden variar con la semilla, hardware y versiones; "
      "pueden persistir sesgos del corpus y del modelo base. La licencia del modelo base no concede "
      "derechos de redistribución sobre el corpus Reuters. No se incluye el corpus en el modelo exportado.")
    p("8. Referencias", "Heading2")
    for title, url in [
        ("Devlin et al. BERT (2018)", "https://arxiv.org/abs/1810.04805"),
        ("BERT-base-cased", "https://huggingface.co/google-bert/bert-base-cased"),
        ("CoNLL-2003 en Hugging Face", "https://huggingface.co/datasets/lhoestq/conll2003"),
        ("CoNLL-2003: tarea original", "https://aclanthology.org/W03-0419/"),
        ("Hugging Face: token classification", "https://huggingface.co/docs/transformers/tasks/token_classification"),
        ("seqeval", "https://github.com/chakki-works/seqeval")]:
        p(f'<link href="{url}" color="#176e99">{title}</link>')

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.8*cm, 1.1*cm, "U2T01 | NER | Jonathan")
        canvas.drawRightString(19.2*cm, 1.1*cm, str(doc.page))

    SimpleDocTemplate(str(ROOT / "reporte_ner.pdf"), rightMargin=1.8*cm, leftMargin=1.8*cm,
                      topMargin=1.5*cm, bottomMargin=1.7*cm).build(story, onFirstPage=footer, onLaterPages=footer)
    return table


if __name__ == "__main__":
    print(build_report().to_string(index=False))
