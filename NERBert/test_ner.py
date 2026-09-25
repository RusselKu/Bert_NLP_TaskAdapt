"""Pruebas de los riesgos principales: alineación, entidades y congelación."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from transformers import BertConfig, BertForTokenClassification
from ner_utils import align_labels, configure_trainable, decode_sequences, entity_metrics, optimizer_groups


class NERTests(unittest.TestCase):
    def test_metric_report_json(self):
        from train_ner import write_json
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.json"
            write_json(path, {"PER": {"support": np.int64(3), "f1": np.float64(0.5)}})
            self.assertEqual(json.loads(path.read_text()), {"PER": {"support": 3, "f1": 0.5}})

    def test_subwords_specials_and_padding(self):
        self.assertEqual(align_labels([None, 0, 1, 1, 2, None, None], [1, 0, 3]),
                         [-100, 1, 0, -100, 3, -100, -100])

    def test_exact_entity_not_token_accuracy(self):
        metrics = entity_metrics([["B-PER", "I-PER", "O"]], [["B-PER", "O", "O"]])
        self.assertEqual(metrics["f1"], 0)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(entity_metrics([["B-PER"]], [["I-PER"]])["f1"], 0)

    def test_ignore_mask(self):
        truth, predicted = decode_sequences(np.array([[0, 1, 0, 1]]),
                                            np.array([[-100, 1, -100, 0]]), ["O", "B-PER"])
        self.assertEqual(truth, [["B-PER", "O"]])
        self.assertEqual(predicted, [["B-PER", "B-PER"]])

    def test_frozen_layers_and_optimizer_coverage(self):
        model = BertForTokenClassification(BertConfig(vocab_size=32, hidden_size=16,
            num_hidden_layers=3, num_attention_heads=2, intermediate_size=32, num_labels=9))
        configure_trainable(model, "partial", 2)
        self.assertFalse(any(p.requires_grad for p in model.bert.embeddings.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.bert.encoder.layer[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.bert.encoder.layer[1:].parameters()))
        groups = optimizer_groups(model, 1e-3, 2e-5, 0.01)
        actual = [id(p) for group in groups for p in group["params"]]
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(expected))
        self.assertEqual({g["lr"] for g in groups}, {1e-3, 2e-5})
        configure_trainable(model, "full")
        self.assertTrue(all(p.requires_grad for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
