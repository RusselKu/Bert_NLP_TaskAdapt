"""Helpers compartidos por el notebook, entrenamiento y pruebas de NER."""
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2


def align_labels(word_ids, labels):
    """Una etiqueta por palabra; especiales y continuaciones se ignoran."""
    previous = None
    aligned = []
    for word_id in word_ids:
        aligned.append(labels[word_id] if word_id is not None and word_id != previous else -100)
        previous = word_id
    return aligned


def decode_sequences(predictions, labels, label_names):
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)
    truth, predicted = [], []
    for pred, gold in zip(predictions, labels):
        truth.append([label_names[int(y)] for y in gold if y != -100])
        predicted.append([label_names[int(p)] for p, y in zip(pred, gold) if y != -100])
    return truth, predicted


def entity_metrics(truth, predicted):
    # CoNLL exporta BIO/IOB2. Una entidad exige span y tipo exactos.
    options = dict(mode="strict", scheme=IOB2, zero_division=0)
    return {
        "f1": f1_score(truth, predicted, **options),
        "precision": precision_score(truth, predicted, **options),
        "recall": recall_score(truth, predicted, **options),
        "accuracy": accuracy_score(truth, predicted),
    }


def configure_trainable(model, method, partial_layers=2):
    if method not in {"partial", "full"}:
        raise ValueError(f"Método desconocido: {method}")
    for parameter in model.parameters():
        parameter.requires_grad = method == "full"
    if method == "partial":
        if not 1 <= partial_layers <= len(model.bert.encoder.layer):
            raise ValueError("Número inválido de capas superiores")
        for layer in model.bert.encoder.layer[-partial_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True


def optimizer_groups(model, head_lr, encoder_lr, weight_decay):
    """Dos LR; cada uno separa bias/LayerNorm para no aplicarles decay."""
    groups = []
    for head in (False, True):
        for decay in (False, True):
            parameters = [p for name, p in model.named_parameters()
                          if p.requires_grad
                          and name.startswith("classifier.") == head
                          and (not (name.endswith("bias") or "LayerNorm.weight" in name)) == decay]
            if parameters:
                groups.append({"params": parameters, "lr": head_lr if head else encoder_lr,
                               "weight_decay": weight_decay if decay else 0.0})
    return groups
