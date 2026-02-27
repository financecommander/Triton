"""
Generate synthetic credit risk training data for Triton.

Creates labeled (Borrower_History_Notes -> Risk_Flag + Audit_Memo) examples
for training the TernaryCreditRiskNet classifier.

Two modes:
  --use-gemini   High-quality labels from Gemini 2.5 Flash (~$2 for 2000)
  (default)      Rule-based labels (free, instant, good for prototyping)

Usage:
    # Rule-based (free, fast)
    python models/scripts/generate_risk_data.py \
        --count 1000 --output data/risk_training.jsonl

    # Gemini-labeled (high quality, ~$2)
    python models/scripts/generate_risk_data.py \
        --count 2000 --output data/risk_training_v2.jsonl --use-gemini

Requires GEMINI_API_KEY in .env or environment for --use-gemini mode.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Synthetic borrower profile components ───────────────────────────

POSITIVE_TRAITS = [
    "excellent track record with multiple successful projects",
    "consistent on-time payments across all prior loans",
    "strong net worth with diversified real estate portfolio",
    "proven experience in commercial real estate development",
    "reliable borrower with 15+ years industry experience",
    "timely repayment history on prior $5M+ facilities",
    "experienced developer with institutional-quality reporting",
    "successful exits on 3 prior multifamily developments",
    "strong sponsor with $50M+ AUM across CRE portfolio",
    "good credit history and stable cash flow from operations",
    "verifiable income sources and ample reserves",
    "seasoned operator with deep market knowledge",
    "consistent debt service coverage ratio above 1.5x",
    "long-standing relationships with institutional lenders",
    "track record of delivering projects on time and under budget",
]

NEUTRAL_TRAITS = [
    "standard borrower with limited operating history",
    "first-time commercial borrower with residential experience",
    "moderate experience in the asset class",
    "borrower has completed one prior project of similar scope",
    "adequate financial statements provided",
    "regional operator expanding into new market",
    "borrower has mixed reviews from prior lenders",
    "limited third-party reporting available",
    "sponsor has adequate but not exceptional liquidity",
    "operating history is under 5 years in CRE",
    "average credit score with no major derogatory marks",
    "borrower is relocating from a different geographic market",
    "limited institutional experience but strong residential background",
    "self-reported financials, pending verification",
]

NEGATIVE_TRAITS = [
    "history of late payments on prior facilities",
    "prior default on $2M construction loan in 2019",
    "bankruptcy filing in 2017, since discharged",
    "foreclosure on prior investment property",
    "delinquent on multiple obligations in past 3 years",
    "poor credit score below 620",
    "failed to complete prior development project",
    "litigation pending related to prior loan default",
    "borrower has significant tax liens outstanding",
    "concern about borrower's ability to fund cost overruns",
    "multiple loan modifications requested in past 2 years",
    "insufficient reserves relative to project scope",
    "prior fraud investigation, case closed without charges",
    "pattern of over-leveraging across portfolio",
    "loss of key tenant at borrower's primary income property",
]

PROJECT_TYPES = [
    "Multifamily", "Office", "Retail", "Industrial",
    "Mixed-Use", "Hotel", "Senior Housing", "Student Housing",
    "Self-Storage", "Medical Office", "Data Center",
    "Single Family Build-to-Rent", "Land Development",
]

LOCATIONS = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "San Francisco", "Austin", "Denver", "Miami", "Seattle",
    "Nashville", "Charlotte", "Dallas", "Atlanta", "Portland",
    "Rural Kentucky", "Small town Iowa", "Suburban Ohio",
    "Tampa", "Raleigh", "Salt Lake City", "San Diego",
    "Minneapolis", "Detroit", "Las Vegas", "Boston",
]


def build_borrower_notes(risk_level: str) -> str:
    """Generate realistic borrower history notes for a given risk level."""
    if risk_level == "Low":
        traits = random.sample(POSITIVE_TRAITS, k=random.randint(2, 4))
        if random.random() < 0.3:
            traits.append(random.choice(NEUTRAL_TRAITS))
    elif risk_level == "Medium":
        pos = random.sample(POSITIVE_TRAITS, k=random.randint(1, 2))
        neu = random.sample(NEUTRAL_TRAITS, k=random.randint(1, 2))
        neg = random.sample(NEGATIVE_TRAITS, k=random.randint(0, 1))
        traits = pos + neu + neg
        random.shuffle(traits)
    else:  # High
        neg = random.sample(NEGATIVE_TRAITS, k=random.randint(2, 4))
        if random.random() < 0.3:
            neg.append(random.choice(NEUTRAL_TRAITS))
        traits = neg

    project = random.choice(PROJECT_TYPES)
    location = random.choice(LOCATIONS)
    amount = random.choice([
        "$500K", "$1.2M", "$2.5M", "$3.5M", "$5M",
        "$7M", "$10M", "$15M", "$25M", "$50M",
    ])

    notes = (
        f"Borrower seeking {amount} for {project} project in {location}. "
        + ". ".join(t.capitalize() for t in traits) + "."
    )
    return notes


def generate_with_gemini(notes: str, api_key: str) -> dict | None:
    """Call Gemini to produce a labeled risk assessment."""
    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai package required. pip install google-genai")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    prompt = (
        "You are an independent credit risk auditor. Review these borrower notes. "
        "Summarize the borrower's 'Execution Risk' in one sentence and assign a "
        "'Risk Flag' (Low, Medium, High). "
        'Return response in JSON format with keys: "Risk_Flag" and "Audit_Memo".\n\n'
        f"Borrower Notes:\n{notes}"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = response.text.strip()

        # Strip markdown code fences
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        result = json.loads(text)
        if "Risk_Flag" in result and "Audit_Memo" in result:
            return result
    except Exception as e:
        print(f"  Gemini error: {e}")
    return None


def generate_rule_based(notes: str, intended_risk: str) -> dict:
    """Deterministic labeling without API call."""
    lower = notes.lower()

    neg_count = sum(1 for kw in [
        "default", "late", "poor", "failed", "bankruptcy",
        "foreclosure", "delinquent", "lien", "litigation", "concern",
        "fraud", "over-leveraging", "modifications", "insufficient",
    ] if kw in lower)

    pos_count = sum(1 for kw in [
        "excellent", "strong", "proven", "reliable", "consistent",
        "timely", "successful", "experienced", "good", "seasoned",
        "verifiable", "institutional", "deep market",
    ] if kw in lower)

    if neg_count >= 2:
        flag = "High"
        memo = "Multiple risk indicators identified including adverse credit events and financial distress."
    elif neg_count == 1 and pos_count <= 1:
        flag = "Medium"
        memo = "Mixed borrower profile with elevated execution risk; additional due diligence warranted."
    elif pos_count >= 2:
        flag = "Low"
        memo = "Strong borrower profile with proven track record and solid financial standing."
    else:
        flag = intended_risk
        memo = {
            "Low": "Adequate borrower profile with no significant risk indicators identified.",
            "Medium": "Moderate risk profile; limited operating history warrants additional review.",
            "High": "Elevated risk profile with significant concerns about borrower capacity.",
        }[intended_risk]

    return {"Risk_Flag": flag, "Audit_Memo": memo}


def main():
    parser = argparse.ArgumentParser(description="Generate credit risk training data")
    parser.add_argument("--count", type=int, default=1000, help="Number of examples")
    parser.add_argument("--output", type=str, default="data/risk_training.jsonl")
    parser.add_argument("--use-gemini", action="store_true",
                        help="Use Gemini API for labels (slower, costs ~$1-2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    api_key = os.getenv("GEMINI_API_KEY", "")

    if args.use_gemini and not api_key:
        print("ERROR: Set GEMINI_API_KEY to use --use-gemini")
        sys.exit(1)

    # Distribution: 40% Low, 35% Medium, 25% High
    risk_distribution = (
        ["Low"] * int(args.count * 0.40)
        + ["Medium"] * int(args.count * 0.35)
        + ["High"] * (args.count - int(args.count * 0.40) - int(args.count * 0.35))
    )
    random.shuffle(risk_distribution)

    examples = []
    gemini_calls = 0

    print(f"Generating {args.count} training examples...")
    print(f"  Mode: {'Gemini API' if args.use_gemini else 'Rule-based (fast)'}")
    print(f"  Output: {args.output}")

    for i, intended_risk in enumerate(risk_distribution):
        notes = build_borrower_notes(intended_risk)

        if args.use_gemini:
            label = generate_with_gemini(notes, api_key)
            gemini_calls += 1
            if label is None:
                label = generate_rule_based(notes, intended_risk)
            if gemini_calls % 10 == 0:
                time.sleep(1)
        else:
            label = generate_rule_based(notes, intended_risk)

        example = {
            "input": notes,
            "output": {
                "Risk_Flag": label["Risk_Flag"],
                "Audit_Memo": label["Audit_Memo"],
            },
        }
        examples.append(example)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.count} generated")

    # Write JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    flags = [ex["output"]["Risk_Flag"] for ex in examples]
    print(f"\nDone! Wrote {len(examples)} examples to {args.output}")
    print(f"  Low:    {flags.count('Low'):4d} ({flags.count('Low')/len(flags)*100:.0f}%)")
    print(f"  Medium: {flags.count('Medium'):4d} ({flags.count('Medium')/len(flags)*100:.0f}%)")
    print(f"  High:   {flags.count('High'):4d} ({flags.count('High')/len(flags)*100:.0f}%)")

    if args.use_gemini:
        print(f"  Gemini API calls: {gemini_calls}")


if __name__ == "__main__":
    main()
