"""Experimentos reproducibles NER. También se invoca desde el notebook."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT.parent / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainerCallback, TrainingArguments, set_seed)
from ner_utils import align_labels, configure_trainable, decode_sequences, entity_metrics, optimizer_groups


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Cannot serialize {type(obj).__name__}")
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=convert), encoding="utf-8")


class EpochTimer(TrainerCallback):
    def __init__(self):
        self.rows = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.started = time.perf_counter()
        self.step = state.global_step

    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        steps = state.global_step - self.step
        self.rows.append({"epoch": state.epoch, "training_seconds": elapsed,
                          "optimizer_steps": steps,
                          "seconds_per_optimizer_step": elapsed / max(1, steps)})


def prepare_data(config, output):
    api = HfApi()
    # Guardar SHAs exactos; para reproducir se usa resolved_config.json.
    config["model_revision"] = config.get("model_revision") or api.model_info(config["model_id"]).sha
    config["dataset_revision"] = config.get("dataset_revision") or api.dataset_info(config["dataset_id"]).sha
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"], use_fast=True)
    raw = load_dataset(config["dataset_id"], revision=config["dataset_revision"])
    names = raw["train"].features["ner_tags"].feature.names
    if set(names) != {"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"}:
        raise ValueError(f"Esquema NER inesperado: {names}")
    if config.get("smoke"):
        for split in raw:
            raw[split] = raw[split].shuffle(seed=config["seed"]).select(range(min(len(raw[split]), 128)))
    stats = {}
    for split, dataset in raw.items():
        for row in dataset:
            if len(row["tokens"]) != len(row["ner_tags"]):
                raise ValueError(f"Etiquetas desalineadas en {split}")
        lengths = tokenizer(list(dataset["tokens"]), is_split_into_words=True, truncation=False)["input_ids"]
        longest = max(map(len, lengths))
        # No descartar palabras ni truncar entidades silenciosamente.
        if longest > config["max_length"]:
            raise ValueError(f"{split}: longitud {longest} > max_length. Implementar ventanas antes de entrenar.")
        stats[split] = {"sentences": len(dataset), "words": sum(len(x) for x in dataset["tokens"]),
                        "max_wordpieces_including_specials": longest, "truncated_sentences": 0,
                        "fingerprint": dataset._fingerprint}

    def tokenize(batch):
        encoded = tokenizer(batch["tokens"], is_split_into_words=True, truncation=False)
        encoded["labels"] = [align_labels(encoded.word_ids(i), tags) for i, tags in enumerate(batch["ner_tags"])]
        return encoded

    encoded = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
    # Batch real con padding; buscar una oración que contenga subpalabras.
    chosen = next((i for i, row in enumerate(encoded["train"])
                   if len(row["input_ids"]) > len(raw["train"][i]["tokens"]) + 2), 0)
    indices = [chosen, (chosen + 1) % len(encoded["train"])]
    batch = collator([encoded["train"][i] for i in indices])
    lines = ["SANITY CHECK: token | etiqueta (primera subpalabra); -100 = ignorar"]
    for j, index in enumerate(indices):
        row = raw["train"][index]
        tokenized = tokenizer(row["tokens"], is_split_into_words=True)
        expected = align_labels(tokenized.word_ids(), row["ner_tags"])
        actual = batch["labels"][j].tolist()
        assert actual[:len(expected)] == expected
        assert all(y == -100 for y in actual[len(expected):])
        assert sum(y != -100 for y in actual) == len(row["tokens"])
        lines.append(f"\nEjemplo train[{index}]")
        for token, label in zip(tokenizer.convert_ids_to_tokens(batch["input_ids"][j].tolist()), actual):
            lines.append(f"{token:24s} {label:5d} {names[label] if label != -100 else 'IGNORE'}")
    sanity = "\n".join(lines)
    print(sanity, flush=True)
    (output / "sanity_check.txt").write_text(sanity, encoding="utf-8")
    write_json(output / "data_audit.json", stats)
    write_json(output / "resolved_config.json", config)
    write_json(output / "labels.json", names)
    return raw, encoded, tokenizer, collator, names


def run_experiment(method, config, encoded, tokenizer, collator, names, output):
    destination = output / method
    destination.mkdir(parents=True, exist_ok=True)
    set_seed(config["seed"])
    model = AutoModelForTokenClassification.from_pretrained(
        config["model_id"], revision=config["model_revision"], num_labels=len(names),
        id2label=dict(enumerate(names)), label2id={name: i for i, name in enumerate(names)},
        attn_implementation="sdpa")
    configure_trainable(model, method, config["partial_layers"])
    initial_head = model.classifier.weight.detach().cpu().clone()
    initial_head_hash = hashlib.sha256(initial_head.numpy().tobytes()).hexdigest()
    counts = {"total": sum(p.numel() for p in model.parameters()),
              "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    optimizer = torch.optim.AdamW(optimizer_groups(model, config["head_lr"], config["encoder_lr"], config["weight_decay"]))
    steps = math.ceil(math.ceil(len(encoded["train"]) / config["batch_size"]) / config["gradient_accumulation_steps"])
    arguments = TrainingArguments(
        output_dir=str(destination / "checkpoints"), num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps", logging_steps=25,
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        save_total_limit=1, save_only_model=True,
        warmup_steps=max(1, math.ceil(steps * config["epochs"] * config["warmup_fraction"])),
        lr_scheduler_type="linear", max_grad_norm=1.0,
        fp16=torch.cuda.is_available(), seed=config["seed"], data_seed=config["seed"],
        report_to="none", dataloader_num_workers=0, disable_tqdm=True,
        dataloader_pin_memory=torch.cuda.is_available(), optim="adamw_torch")
    timer = EpochTimer()

    def metrics(prediction):
        truth, predicted = decode_sequences(prediction.predictions, prediction.label_ids, names)
        return entity_metrics(truth, predicted)

    trainer = Trainer(model=model, args=arguments, train_dataset=encoded["train"],
                      eval_dataset=encoded["validation"], processing_class=tokenizer,
                      data_collator=collator, compute_metrics=metrics, callbacks=[timer],
                      optimizers=(optimizer, None))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    training = trainer.train()
    elapsed = time.perf_counter() - started
    if torch.equal(initial_head, model.classifier.weight.detach().cpu()):
        raise RuntimeError("La cabeza no cambió: revisar pasos omitidos por overflow o LR nulo")
    validation = trainer.evaluate(metric_key_prefix="validation")
    trainer.save_model(str(destination / "model"))
    tokenizer.save_pretrained(destination / "model")
    result = {"method": method, "parameters": counts, "training_wall_seconds": elapsed,
              "initial_head_sha256": initial_head_hash, "head_weights_changed": True,
              "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
              "training_metrics": training.metrics, "validation_metrics": validation,
              "best_checkpoint": trainer.state.best_model_checkpoint,
              "epoch_timing": timer.rows, "history": trainer.state.log_history}
    write_json(destination / "result.json", result)
    del trainer, model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def evaluate_test(method, config, encoded, tokenizer, collator, names, output):
    model = AutoModelForTokenClassification.from_pretrained(output / method / "model")
    args = TrainingArguments(output_dir=str(output / method / "evaluation"),
                             per_device_eval_batch_size=config["eval_batch_size"],
                             report_to="none", disable_tqdm=True,
                             fp16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=args, processing_class=tokenizer, data_collator=collator)
    prediction = trainer.predict(encoded["test"])
    truth, predicted = decode_sequences(prediction.predictions, prediction.label_ids, names)
    metrics = entity_metrics(truth, predicted)
    metrics.update(prediction.metrics)
    report = classification_report(truth, predicted, mode="strict", scheme=IOB2, zero_division=0, output_dict=True)
    write_json(output / method / "test_metrics.json", metrics)
    write_json(output / method / "test_entity_report.json", report)
    # Predicciones completas permiten reproducir las métricas sin GPU; no copiar el corpus.
    write_json(output / method / "test_predictions.json", {"truth": truth, "predicted": predicted})
    errors = [{"test_index": i, "gold": gold, "predicted": pred}
              for i, (gold, pred) in enumerate(zip(truth, predicted)) if gold != pred][:20]
    write_json(output / method / "error_examples.json", errors)
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main(config_path=ROOT / "config.json", smoke=False):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config["smoke"] = smoke
    if smoke:
        config["epochs"] = 1
    output = ROOT / ("smoke_results" if smoke else "results")
    output.mkdir(parents=True, exist_ok=True)
    # Evitar sobreescribir experimentos terminados accidentalmente.
    if (output / "comparison.json").exists():
        raise FileExistsError(f"Resultados ya existentes en {output}; usar otra copia del proyecto para repetir.")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    set_seed(config["seed"])
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "torch": torch.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    print(json.dumps(environment, indent=2), flush=True)
    write_json(output / "environment.json", environment)
    (output / "requirements-lock.txt").write_text(subprocess.check_output(
        [os.sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    raw, encoded, tokenizer, collator, names = prepare_data(config, output)
    results = [run_experiment(method, config, encoded, tokenizer, collator, names, output)
               for method in config["methods"]]
    # Decisión tomada ANTES de observar test. Empate exacto: menos parámetros.
    winner = max(results, key=lambda r: (r["validation_metrics"]["validation_f1"], -r["parameters"]["trainable"]))["method"]
    write_json(output / "selection.json", {"winner": winner, "criterion": "validation strict entity micro F1; exact tie: fewer trainable parameters"})
    for result in results:
        result["test_metrics"] = evaluate_test(result["method"], config, encoded, tokenizer, collator, names, output)
    write_json(output / "comparison.json", {"winner": winner, "smoke": smoke, "runs": results})
    print(f"Experimentos terminados. Ganador por validación: {winner}. Resultados: {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--smoke", action="store_true", help="128 oraciones por split; NO es un resultado de entrega")
    args = parser.parse_args()
    main(args.config, args.smoke)
