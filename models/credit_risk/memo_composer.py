"""
Compositional Audit Memo Generator for Credit Risk Classification

Replaces static 3-template MEMO_TEMPLATES with a lightweight, on-device
memo composer that extracts risk factors from borrower input text and
assembles contextual audit narratives.

Architecture:
  1. Extract risk factors from input via keyword/pattern matching
  2. Select memo fragments based on (risk_label × detected_factors)
  3. Compose a 1-2 sentence audit memo with appropriate framing

This keeps inference fully on-device (no LLM, no decoder head) while
producing memos closer in quality to the Gemini-labeled training data.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Risk Factor Taxonomy ────────────────────────────────────────────

@dataclass
class ExtractedFactors:
    """Risk factors extracted from borrower history text."""
    project_type: Optional[str] = None
    location: Optional[str] = None
    loan_amount: Optional[str] = None
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    neutral: list[str] = field(default_factory=list)


# Patterns for extraction (ordered by specificity)
PROJECT_TYPES = [
    "student housing", "senior housing", "multifamily", "mixed-use",
    "office", "retail", "industrial", "hotel", "data center",
    "warehouse", "self-storage", "medical office", "life science",
]

STRENGTH_SIGNALS = {
    r"experienced developer": "experienced developer with proven track record",
    r"institutional[- ]quality": "institutional-quality reporting and controls",
    r"strong sponsor": "strong sponsorship backing",
    r"\$\d+m\+ aum": "substantial asset base under management",
    r"ample reserves": "ample financial reserves",
    r"timely repayment": "timely repayment history on prior facilities",
    r"proven experience": "proven commercial real estate experience",
    r"strong net worth": "strong net worth with diversified holdings",
    r"stable cash flow": "stable cash flow from existing operations",
    r"good credit history": "good credit history",
    r"verifiable income": "verifiable income sources",
    r"strong dscr|debt service coverage ratio above": "strong debt service coverage",
    r"diversified revenue": "diversified revenue streams",
    r"repeat borrower": "established relationship as repeat borrower",
}

WEAKNESS_SIGNALS = {
    r"poor credit score|credit score below 620": "poor credit score",
    r"bankruptcy": "prior bankruptcy filing",
    r"late payment|delinquent": "history of late payments",
    r"fraud investigation": "prior fraud investigation",
    r"prior default": "prior loan default",
    r"limited operating history": "limited direct operating history",
    r"no prior.*experience": "no prior experience in asset class",
    r"litigation|legal action|lawsuit": "ongoing or prior litigation",
    r"declining revenue|revenue decline": "declining revenue trend",
    r"over-?leveraged|high leverage": "elevated leverage profile",
    r"mixed reviews from.*lenders": "mixed feedback from prior lenders",
    r"incomplete financials|missing.*documentation": "incomplete financial documentation",
    r"environmental": "environmental compliance concerns",
}

NEUTRAL_SIGNALS = {
    r"average credit score": "average credit profile with no major derogatory marks",
    r"first[- ]time.*borrower": "first-time borrower in this asset class",
    r"regional operator": "regional operator scope",
}

# Location patterns
LOCATION_RE = re.compile(
    r"(?:in|near)\s+([A-Z][a-zA-Z\s]+(?:,\s*[A-Z]{2})?)",
    re.IGNORECASE,
)

LOAN_AMOUNT_RE = re.compile(
    r"\$[\d,.]+[MmKk]?\b",
)


# ── Extraction ──────────────────────────────────────────────────────

def extract_risk_factors(text: str) -> ExtractedFactors:
    """Extract structured risk factors from borrower history text."""
    factors = ExtractedFactors()
    text_lower = text.lower()

    # Project type
    for pt in PROJECT_TYPES:
        if pt in text_lower:
            factors.project_type = pt.title()
            break

    # Location
    loc_match = LOCATION_RE.search(text)
    if loc_match:
        factors.location = loc_match.group(1).strip().rstrip(".")

    # Loan amount
    amt_match = LOAN_AMOUNT_RE.search(text)
    if amt_match:
        factors.loan_amount = amt_match.group(0)

    # Strengths
    for pattern, description in STRENGTH_SIGNALS.items():
        if re.search(pattern, text_lower):
            factors.strengths.append(description)

    # Weaknesses
    for pattern, description in WEAKNESS_SIGNALS.items():
        if re.search(pattern, text_lower):
            factors.weaknesses.append(description)

    # Neutral
    for pattern, description in NEUTRAL_SIGNALS.items():
        if re.search(pattern, text_lower):
            factors.neutral.append(description)

    return factors


# ── Memo Composition ────────────────────────────────────────────────

def compose_audit_memo(
    risk_label: str,
    factors: ExtractedFactors,
    confidence: float,
) -> str:
    """Compose a contextual audit memo from risk label and extracted factors.

    Returns a 1-3 sentence narrative suitable for credit committee review.
    """
    parts: list[str] = []

    # Build the project context prefix
    ctx = _project_context(factors)

    if risk_label == "Low":
        parts.append(_compose_low(factors, ctx))
    elif risk_label == "Medium":
        parts.append(_compose_medium(factors, ctx))
    else:  # High
        parts.append(_compose_high(factors, ctx))

    # Confidence qualifier
    if confidence < 0.6:
        parts.append(
            "Classification confidence is moderate; manual review recommended."
        )
    elif confidence < 0.75 and risk_label == "High":
        parts.append(
            "Recommend senior credit officer review prior to final determination."
        )

    return " ".join(parts)


def _project_context(factors: ExtractedFactors) -> str:
    """Build a brief project context string."""
    segments = []
    if factors.loan_amount:
        segments.append(factors.loan_amount)
    if factors.project_type:
        segments.append(f"{factors.project_type} project")
    if factors.location:
        segments.append(f"in {factors.location}")

    if segments:
        return " ".join(segments)
    return "proposed project"


def _compose_low(factors: ExtractedFactors, ctx: str) -> str:
    """Compose memo for Low risk classification."""
    if factors.strengths:
        lead_strength = factors.strengths[0]
        if len(factors.strengths) > 1:
            supporting = factors.strengths[1]
            return (
                f"The {ctx} benefits from {lead_strength}, "
                f"further supported by {supporting}, "
                f"resulting in a favorable risk profile."
            )
        return (
            f"The {ctx} presents low execution risk, "
            f"primarily due to {lead_strength}."
        )
    return (
        f"The {ctx} presents a strong borrower profile "
        f"with no significant risk indicators identified."
    )


def _compose_medium(factors: ExtractedFactors, ctx: str) -> str:
    """Compose memo for Medium risk classification."""
    if factors.weaknesses and factors.strengths:
        weakness = factors.weaknesses[0]
        mitigation = factors.strengths[0]
        return (
            f"Execution risk for the {ctx} is elevated due to {weakness}, "
            f"though partially mitigated by {mitigation}. "
            f"Additional due diligence is warranted."
        )
    if factors.weaknesses:
        weakness = factors.weaknesses[0]
        return (
            f"The {ctx} carries moderate risk due to {weakness}. "
            f"Standard enhanced monitoring recommended."
        )
    if factors.neutral:
        note = factors.neutral[0]
        return (
            f"The {ctx} reflects a mixed risk profile ({note}). "
            f"Enhanced due diligence recommended prior to commitment."
        )
    return (
        f"The {ctx} presents a mixed borrower profile; "
        f"elevated execution risk warrants additional due diligence."
    )


def _compose_high(factors: ExtractedFactors, ctx: str) -> str:
    """Compose memo for High risk classification."""
    if len(factors.weaknesses) >= 2:
        w1, w2 = factors.weaknesses[0], factors.weaknesses[1]
        return (
            f"The {ctx} presents significant risk, "
            f"driven by {w1} and {w2}. "
            f"Credit committee escalation required."
        )
    if factors.weaknesses:
        weakness = factors.weaknesses[0]
        return (
            f"High execution risk identified for the {ctx} "
            f"due to {weakness}. "
            f"Recommend declining or requiring substantial risk mitigants."
        )
    return (
        f"Significant risk factors identified for the {ctx}, "
        f"including adverse credit events and financial distress indicators. "
        f"Credit committee escalation required."
    )
