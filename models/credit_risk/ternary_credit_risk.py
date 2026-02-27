"""
Ternary Credit Risk Classification Model

A lightweight ternary neural network for classifying borrower risk from
text-based history notes. Designed for on-device, zero-cost inference
as a drop-in replacement for cloud LLM risk analysis.

Architecture:
  Embedding(vocab, 128) → TernaryLinear(128, 256) → TernaryLinear(256, 128) →
  TernaryLinear(128, 3)  [Low / Medium / High]

  + Separate memo generation head for audit narrative.

Input:  Borrower history notes (tokenized text)
Output: Risk flag (Low/Medium/High) + Audit memo
"""

import json
import re
import torch
import torch.nn as nn
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.pytorch.ternary_tensor import TernaryLinear


# ── Tokenizer ───────────────────────────────────────────────────────

class CreditRiskTokenizer:
    """Simple word-level tokenizer for borrower history notes.

    Builds a vocabulary from training data. Words not seen during
    training are mapped to <UNK>. Sequences are padded/truncated
    to max_length.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, max_length: int = 128, min_freq: int = 2):
        self.max_length = max_length
        self.min_freq = min_freq
        self.word2idx: dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word: dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.word_freq: dict[str, int] = {}
        self.vocab_size = 2

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\.\,\$\%\-]", " ", text)
        return text.split()

    def build_vocab(self, texts: list[str]):
        """Build vocabulary from a list of texts."""
        self.word_freq = {}
        for text in texts:
            for word in self._tokenize(text):
                self.word_freq[word] = self.word_freq.get(word, 0) + 1

        for word, freq in sorted(self.word_freq.items()):
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to padded tensor of token IDs."""
        tokens = self._tokenize(text)
        ids = [self.word2idx.get(w, 1) for w in tokens]  # 1 = UNK
        # Truncate or pad
        ids = ids[: self.max_length]
        ids += [0] * (self.max_length - len(ids))  # 0 = PAD
        return torch.tensor(ids, dtype=torch.long)

    def save(self, path: str):
        data = {
            "max_length": self.max_length,
            "min_freq": self.min_freq,
            "word2idx": self.word2idx,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "CreditRiskTokenizer":
        with open(path, "r") as f:
            data = json.load(f)
        tok = cls(max_length=data["max_length"], min_freq=data["min_freq"])
        tok.word2idx = data["word2idx"]
        tok.idx2word = {int(v): k for k, v in tok.word2idx.items()}
        tok.vocab_size = len(tok.word2idx)
        return tok


# ── Risk flag labels ────────────────────────────────────────────────

RISK_LABELS = ["Low", "Medium", "High"]


# ── Memo templates (used when model doesn't generate free-form) ─────

MEMO_TEMPLATES = {
    "Low": "Strong borrower profile with no significant risk indicators identified.",
    "Medium": "Mixed borrower profile; elevated execution risk warrants additional due diligence.",
    "High": "Significant risk factors identified including adverse credit events and financial distress indicators.",
}


# ── Model ───────────────────────────────────────────────────────────

class TernaryCreditRiskNet(nn.Module):
    """Ternary neural network for credit risk classification.

    Uses TernaryLinear layers from Triton's backend for 2-bit packed
    inference. Embedding layer is kept in full precision for
    vocabulary quality.

    Args:
        vocab_size: Vocabulary size.
        embed_dim: Embedding dimension (default 128).
        hidden_dim: Hidden layer dimension (default 256).
        num_classes: Number of risk classes (default 3: Low/Medium/High).
        dropout: Dropout probability (default 0.3).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Embedding (full precision — small, quality-sensitive)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Ternary classification layers
        self.fc1 = TernaryLinear(embed_dim, hidden_dim)
        self.fc2 = TernaryLinear(hidden_dim, hidden_dim // 2)
        self.fc3 = TernaryLinear(hidden_dim // 2, num_classes)

        # Normalization and regularization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs.

        Returns:
            (batch, num_classes) logits.
        """
        # Embed and mean-pool over sequence
        x = self.embedding(input_ids)  # (B, L, E)
        # Mask padding tokens (id=0) before pooling
        mask = (input_ids != 0).unsqueeze(-1).float()  # (B, L, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, E)

        # Ternary classifier
        x = self.act(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.act(self.norm2(self.fc2(x)))
        x = self.dropout(x)
        logits = self.fc3(x)  # (B, 3)

        return logits

    def predict(self, input_ids: torch.Tensor) -> list[dict]:
        """Run inference and return structured risk assessments.

        Args:
            input_ids: (batch, seq_len) token IDs.

        Returns:
            List of {"Risk_Flag": str, "Audit_Memo": str} dicts.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

        results = []
        for i in range(preds.size(0)):
            flag = RISK_LABELS[preds[i].item()]
            confidence = probs[i, preds[i]].item()
            memo = MEMO_TEMPLATES[flag]
            if confidence < 0.6:
                memo += " (Note: classification confidence is moderate; manual review recommended.)"
            results.append({
                "Risk_Flag": flag,
                "Audit_Memo": memo,
                "confidence": round(confidence, 3),
            })
        return results


def quantize_model_weights(model: nn.Module, threshold: float = 0.05):
    """Re-quantize ternary layer weights during training."""
    from backend.pytorch.ternary_tensor import TernaryTensor

    for module in model.modules():
        if isinstance(module, TernaryLinear):
            with torch.no_grad():
                module.weight.data = TernaryTensor.ternarize(
                    module.weight.data, threshold
                )
