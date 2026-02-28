"""
Run credit risk inference using a trained Triton ternary model.

Usage:
    # Interactive mode
    python models/scripts/infer_credit_risk.py \
        --checkpoint checkpoints/credit_risk_best.pth \
        --tokenizer checkpoints/tokenizer.json

    # Batch mode (from file)
    python models/scripts/infer_credit_risk.py \
        --checkpoint checkpoints/credit_risk_best.pth \
        --tokenizer checkpoints/tokenizer.json \
        --input data/test_leads.jsonl \
        --output data/results.jsonl

    # Single text from command line
    python models/scripts/infer_credit_risk.py \
        --checkpoint checkpoints/credit_risk_best.pth \
        --tokenizer checkpoints/tokenizer.json \
        --text "Borrower seeking $5M for Multifamily project. Excellent track record."
"""

import argparse
import json
import sys
import time
import torch
from pathlib import Path

# Add project root

from models.credit_risk.ternary_credit_risk import (
    TernaryCreditRiskNet,
    CreditRiskTokenizer,
)


def load_model(checkpoint_path: str, tokenizer_path: str, device: str = "cpu"):
    """Load trained model and tokenizer from checkpoint."""
    tokenizer = CreditRiskTokenizer.load(tokenizer_path)

    ckpt = torch.load(checkpoint_path, map_location=device)
    vocab_size = ckpt.get("vocab_size", tokenizer.vocab_size)

    model = TernaryCreditRiskNet(vocab_size=vocab_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer


def analyze_text(model, tokenizer, text: str, device: str = "cpu") -> dict:
    """Run risk analysis on a single text."""
    input_ids = tokenizer.encode(text).unsqueeze(0).to(device)

    t0 = time.time()
    results = model.predict(input_ids)
    elapsed_ms = (time.time() - t0) * 1000

    result = results[0]
    result["latency_ms"] = round(elapsed_ms, 1)
    return result


def main():
    parser = argparse.ArgumentParser(description="Ternary Credit Risk Inference")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON path")
    parser.add_argument("--text", type=str, default=None, help="Single text to analyze")
    parser.add_argument("--input", type=str, default=None, help="JSONL input file")
    parser.add_argument("--output", type=str, default=None, help="JSONL output file")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, args.device)

    num_params = sum(p.numel() for p in model.parameters())
    ternary_params = sum(
        p.numel() for n, p in model.named_parameters() if "fc" in n and "weight" in n
    )
    print(f"Model loaded: {num_params:,} params ({ternary_params:,} ternary)")
    print(f"Ternary size: ~{ternary_params * 2.5 / 8 / 1024:.1f} KB")
    print()

    if args.text:
        # Single text mode
        result = analyze_text(model, tokenizer, args.text, args.device)
        print(f"Risk Flag:  {result['Risk_Flag']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Audit Memo: {result['Audit_Memo']}")
        print(f"Latency:    {result['latency_ms']:.1f}ms")

    elif args.input:
        # Batch mode
        results = []
        with open(args.input, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print(f"Analyzing {len(lines)} leads...")
        t0 = time.time()

        for i, line in enumerate(lines):
            obj = json.loads(line.strip())
            text = obj.get("input", obj.get("text", ""))
            result = analyze_text(model, tokenizer, text, args.device)
            result["input"] = text
            results.append(result)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(lines)} analyzed")

        elapsed = time.time() - t0

        # Write output
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"\nResults saved to {args.output}")

        # Summary
        flags = [r["Risk_Flag"] for r in results]
        print(f"\nAnalyzed {len(results)} leads in {elapsed:.1f}s ({elapsed/len(results)*1000:.0f}ms/lead)")
        print(f"  Low:    {flags.count('Low'):4d} ({flags.count('Low')/len(flags)*100:.0f}%)")
        print(f"  Medium: {flags.count('Medium'):4d} ({flags.count('Medium')/len(flags)*100:.0f}%)")
        print(f"  High:   {flags.count('High'):4d} ({flags.count('High')/len(flags)*100:.0f}%)")

    else:
        # Interactive mode
        print("Interactive credit risk analysis (type 'quit' to exit)")
        print("-" * 60)
        while True:
            try:
                text = input("\nBorrower notes> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue

            result = analyze_text(model, tokenizer, text, args.device)
            print(f"  Risk Flag:  {result['Risk_Flag']}")
            print(f"  Confidence: {result['confidence']:.1%}")
            print(f"  Audit Memo: {result['Audit_Memo']}")
            print(f"  Latency:    {result['latency_ms']:.1f}ms")


if __name__ == "__main__":
    main()
