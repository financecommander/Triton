"""
Unit tests for Ternary Credit Risk model.

Tests the TernaryCreditRiskNet architecture, CreditRiskTokenizer,
training utilities, and model zoo registration.
"""

import sys
import os
import json
import tempfile
import unittest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import torch
import torch.nn as nn
from backend.pytorch.ternary_tensor import TernaryLinear


class TestCreditRiskTokenizer(unittest.TestCase):
    """Test the CreditRiskTokenizer."""

    def setUp(self):
        from models.credit_risk.ternary_credit_risk import CreditRiskTokenizer
        self.tokenizer = CreditRiskTokenizer(max_length=32, min_freq=1)
        self.sample_texts = [
            "Borrower seeking $5M for Multifamily project in New York.",
            "Strong track record with multiple successful projects.",
            "Prior default on construction loan in 2019.",
            "Excellent credit history and timely repayment history.",
        ]
        self.tokenizer.build_vocab(self.sample_texts)

    def test_vocab_builds(self):
        """Tokenizer builds vocabulary from training texts."""
        # PAD + UNK + unique words from samples
        self.assertGreater(self.tokenizer.vocab_size, 2)
        self.assertIn("<PAD>", self.tokenizer.word2idx)
        self.assertIn("<UNK>", self.tokenizer.word2idx)

    def test_encode_returns_tensor(self):
        """Encode returns a LongTensor of correct length."""
        encoded = self.tokenizer.encode("Borrower seeking $5M for project")
        self.assertIsInstance(encoded, torch.Tensor)
        self.assertEqual(encoded.dtype, torch.long)
        self.assertEqual(len(encoded), 32)

    def test_encode_pads_short_text(self):
        """Short texts are padded to max_length."""
        encoded = self.tokenizer.encode("hello")
        # Most of the tensor should be 0 (PAD)
        self.assertEqual(encoded[1:].sum().item(), 0)  # After first token, all PAD

    def test_encode_truncates_long_text(self):
        """Long texts are truncated to max_length."""
        long_text = " ".join(["word"] * 100)
        encoded = self.tokenizer.encode(long_text)
        self.assertEqual(len(encoded), 32)

    def test_unknown_words_get_unk(self):
        """Unknown words are mapped to UNK token (id=1)."""
        encoded = self.tokenizer.encode("xyzzy_unknown_word")
        self.assertEqual(encoded[0].item(), 1)  # UNK

    def test_save_and_load(self):
        """Tokenizer can be saved and loaded."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            temp_path = f.name

        try:
            self.tokenizer.save(temp_path)

            from models.credit_risk.ternary_credit_risk import CreditRiskTokenizer
            loaded = CreditRiskTokenizer.load(temp_path)

            self.assertEqual(loaded.vocab_size, self.tokenizer.vocab_size)
            self.assertEqual(loaded.max_length, self.tokenizer.max_length)

            # Encode same text with both — should match
            text = "Borrower seeking $5M for project"
            orig = self.tokenizer.encode(text)
            reloaded = loaded.encode(text)
            self.assertTrue(torch.equal(orig, reloaded))
        finally:
            os.remove(temp_path)


class TestTernaryCreditRiskNet(unittest.TestCase):
    """Test the TernaryCreditRiskNet model architecture."""

    def setUp(self):
        from models.credit_risk.ternary_credit_risk import TernaryCreditRiskNet
        self.model = TernaryCreditRiskNet(
            vocab_size=100, embed_dim=32, hidden_dim=64, dropout=0.0
        )

    def test_output_shape(self):
        """Model outputs (batch, 3) logits."""
        x = torch.randint(0, 100, (4, 32))  # batch=4, seq=32
        logits = self.model(x)
        self.assertEqual(logits.shape, (4, 3))

    def test_single_input(self):
        """Model handles single-example batch."""
        x = torch.randint(0, 100, (1, 32))
        logits = self.model(x)
        self.assertEqual(logits.shape, (1, 3))

    def test_embedding_is_full_precision(self):
        """Embedding layer is not ternary (quality-sensitive)."""
        self.assertIsInstance(self.model.embedding, nn.Embedding)
        self.assertNotIsInstance(self.model.embedding, TernaryLinear)

    def test_fc_layers_are_ternary(self):
        """Classification layers use TernaryLinear."""
        self.assertIsInstance(self.model.fc1, TernaryLinear)
        self.assertIsInstance(self.model.fc2, TernaryLinear)
        self.assertIsInstance(self.model.fc3, TernaryLinear)

    def test_padding_mask_works(self):
        """Padding tokens (id=0) are masked during mean pooling."""
        # Create input with lots of padding
        x = torch.zeros(1, 32, dtype=torch.long)
        x[0, 0] = 5  # Only one real token
        x[0, 1] = 10  # And one more

        # Should not error and should produce finite output
        logits = self.model(x)
        self.assertTrue(torch.isfinite(logits).all())

    def test_all_padding_input(self):
        """Model handles all-padding input gracefully."""
        x = torch.zeros(1, 32, dtype=torch.long)  # All PAD
        logits = self.model(x)
        self.assertTrue(torch.isfinite(logits).all())

    def test_predict_returns_structured(self):
        """predict() returns list of dicts with Risk_Flag and Audit_Memo."""
        x = torch.randint(0, 100, (3, 32))
        results = self.model.predict(x)

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn("Risk_Flag", r)
            self.assertIn("Audit_Memo", r)
            self.assertIn("confidence", r)
            self.assertIn(r["Risk_Flag"], ["Low", "Medium", "High"])
            self.assertGreater(r["confidence"], 0.0)
            self.assertLessEqual(r["confidence"], 1.0)

    def test_parameter_count(self):
        """Model has a reasonable number of parameters."""
        num_params = sum(p.numel() for p in self.model.parameters())
        # Should be small — this is a lightweight classifier
        self.assertGreater(num_params, 1000)
        self.assertLess(num_params, 5_000_000)

    def test_ternary_weight_count(self):
        """Most parameters are in ternary layers."""
        total = sum(p.numel() for p in self.model.parameters())
        ternary = sum(
            p.numel() for n, p in self.model.named_parameters()
            if "fc" in n and "weight" in n
        )
        # At least 30% of params should be ternary
        self.assertGreater(ternary / total, 0.3)


class TestQuantizeModelWeights(unittest.TestCase):
    """Test periodic weight requantization during training."""

    def test_quantize_forces_ternary(self):
        """quantize_model_weights forces fc weights to {-1, 0, 1}."""
        from models.credit_risk.ternary_credit_risk import (
            TernaryCreditRiskNet,
            quantize_model_weights,
        )
        model = TernaryCreditRiskNet(vocab_size=50, embed_dim=16, hidden_dim=32)

        # Before quantization, weights are continuous
        quantize_model_weights(model)

        # After, fc weights should be in {-1, 0, 1}
        for name, param in model.named_parameters():
            if "fc" in name and "weight" in name:
                unique = torch.unique(param.data)
                for val in unique:
                    self.assertIn(val.item(), [-1.0, 0.0, 1.0],
                                  f"Non-ternary value {val} in {name}")


class TestModelZooRegistration(unittest.TestCase):
    """Test that credit risk model is registered in model zoo."""

    def test_credit_risk_in_zoo(self):
        """ternary_credit_risk is listed in MODEL_ZOO."""
        from models.model_zoo import MODEL_ZOO, list_models
        self.assertIn('ternary_credit_risk', MODEL_ZOO)
        self.assertIn('ternary_credit_risk', list_models())

    def test_credit_risk_metadata(self):
        """Credit risk model has correct metadata."""
        from models.model_zoo import get_model_info
        info = get_model_info('ternary_credit_risk')
        self.assertEqual(info['architecture'], 'ternary_credit_risk')
        self.assertEqual(info['num_classes'], 3)
        self.assertEqual(info['dataset'], 'credit_risk_borrower_notes')
        self.assertIn('extra', info)
        self.assertEqual(info['extra']['labels'], ['Low', 'Medium', 'High'])


class TestTrainingDataFormat(unittest.TestCase):
    """Test that training data exists and has correct format."""

    DATA_PATH = os.path.join(
        os.path.dirname(__file__), '../../data/risk_training.jsonl'
    )

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__), '../../data/risk_training.jsonl')),
        "Training data not generated yet"
    )
    def test_data_file_exists(self):
        """Training data file exists."""
        self.assertTrue(os.path.exists(self.DATA_PATH))

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__), '../../data/risk_training.jsonl')),
        "Training data not generated yet"
    )
    def test_data_format(self):
        """Each line is valid JSONL with input and output fields."""
        with open(self.DATA_PATH, 'r') as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break  # Just check first 10
                obj = json.loads(line.strip())
                self.assertIn("input", obj)
                self.assertIn("output", obj)
                self.assertIn("Risk_Flag", obj["output"])
                self.assertIn("Audit_Memo", obj["output"])
                self.assertIn(obj["output"]["Risk_Flag"], ["Low", "Medium", "High"])

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__), '../../data/risk_training.jsonl')),
        "Training data not generated yet"
    )
    def test_data_distribution(self):
        """Training data has all three risk classes."""
        flags = set()
        with open(self.DATA_PATH, 'r') as f:
            for line in f:
                obj = json.loads(line.strip())
                flags.add(obj["output"]["Risk_Flag"])
        self.assertEqual(flags, {"Low", "Medium", "High"})


class TestEndToEnd(unittest.TestCase):
    """End-to-end test: tokenize -> model -> prediction."""

    def test_full_pipeline(self):
        """Full pipeline: text -> tokenize -> model -> risk flag."""
        from models.credit_risk.ternary_credit_risk import (
            TernaryCreditRiskNet,
            CreditRiskTokenizer,
        )

        # Build tokenizer on sample data
        texts = [
            "Borrower seeking $5M for Multifamily project. Excellent track record.",
            "Prior default on loan. Poor credit history.",
            "First-time borrower with moderate experience.",
        ]
        tokenizer = CreditRiskTokenizer(max_length=32, min_freq=1)
        tokenizer.build_vocab(texts)

        # Create model
        model = TernaryCreditRiskNet(
            vocab_size=tokenizer.vocab_size,
            embed_dim=32, hidden_dim=64, dropout=0.0
        )

        # Encode and predict
        input_ids = torch.stack([tokenizer.encode(t) for t in texts])
        results = model.predict(input_ids)

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn(r["Risk_Flag"], ["Low", "Medium", "High"])
            self.assertIsInstance(r["Audit_Memo"], str)
            self.assertGreater(len(r["Audit_Memo"]), 10)


if __name__ == '__main__':
    unittest.main()
