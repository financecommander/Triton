"""Tests for compositional audit memo generation."""

import pytest
from models.credit_risk.memo_composer import (
    ExtractedFactors,
    extract_risk_factors,
    compose_audit_memo,
)


class TestRiskFactorExtraction:
    """Test risk factor extraction from borrower text."""

    def test_extracts_project_type(self):
        text = "Borrower seeking $5M for Multifamily project in Nashville."
        factors = extract_risk_factors(text)
        assert factors.project_type == "Multifamily"

    def test_extracts_location(self):
        text = "Borrower seeking $5M for Hotel project in Miami."
        factors = extract_risk_factors(text)
        assert factors.location == "Miami"

    def test_extracts_loan_amount(self):
        text = "Borrower seeking $1.2M for Student Housing project."
        factors = extract_risk_factors(text)
        assert factors.loan_amount == "$1.2M"

    def test_extracts_strength_signals(self):
        text = (
            "Experienced developer with institutional-quality reporting. "
            "Ample reserves and verifiable income sources."
        )
        factors = extract_risk_factors(text)
        assert len(factors.strengths) >= 3
        assert any("experienced" in s for s in factors.strengths)
        assert any("institutional" in s for s in factors.strengths)

    def test_extracts_weakness_signals(self):
        text = (
            "Poor credit score below 620. Bankruptcy filing in 2022. "
            "Prior default on a $2M construction loan."
        )
        factors = extract_risk_factors(text)
        assert len(factors.weaknesses) >= 3
        assert any("credit score" in w for w in factors.weaknesses)
        assert any("bankruptcy" in w for w in factors.weaknesses)

    def test_extracts_neutral_signals(self):
        text = "Average credit score with no major derogatory marks."
        factors = extract_risk_factors(text)
        assert len(factors.neutral) >= 1

    def test_empty_text(self):
        factors = extract_risk_factors("")
        assert factors.project_type is None
        assert factors.location is None
        assert factors.strengths == []
        assert factors.weaknesses == []

    def test_mixed_signals(self):
        text = (
            "Borrower seeking $500K for Hotel project in Miami. "
            "Strong sponsor with $50m+ aum across cre portfolio. "
            "Late payment history on multiple facilities."
        )
        factors = extract_risk_factors(text)
        assert factors.project_type == "Hotel"
        assert factors.location == "Miami"
        assert len(factors.strengths) >= 1
        assert len(factors.weaknesses) >= 1


class TestMemoComposition:
    """Test audit memo composition from risk label + factors."""

    def test_low_risk_with_strengths(self):
        factors = ExtractedFactors(
            project_type="Multifamily",
            location="Nashville",
            loan_amount="$5M",
            strengths=["experienced developer with proven track record"],
        )
        memo = compose_audit_memo("Low", factors, confidence=0.92)
        assert "low execution risk" in memo.lower() or "benefits from" in memo.lower()
        assert "$5M" in memo
        assert "experienced" in memo.lower()

    def test_low_risk_multiple_strengths(self):
        factors = ExtractedFactors(
            project_type="Office",
            strengths=[
                "experienced developer with proven track record",
                "stable cash flow from existing operations",
            ],
        )
        memo = compose_audit_memo("Low", factors, confidence=0.88)
        assert "experienced" in memo.lower()
        assert "cash flow" in memo.lower()

    def test_medium_risk_with_weakness_and_mitigation(self):
        factors = ExtractedFactors(
            project_type="Data Center",
            location="Nashville",
            weaknesses=["limited direct operating history"],
            strengths=["strong sponsorship backing"],
        )
        memo = compose_audit_memo("Medium", factors, confidence=0.72)
        assert "limited" in memo.lower()
        assert "sponsorship" in memo.lower() or "mitigated" in memo.lower()

    def test_high_risk_multiple_weaknesses(self):
        factors = ExtractedFactors(
            project_type="Hotel",
            location="Boston",
            weaknesses=[
                "poor credit score",
                "prior bankruptcy filing",
            ],
        )
        memo = compose_audit_memo("High", factors, confidence=0.85)
        assert "credit score" in memo.lower()
        assert "bankruptcy" in memo.lower()
        assert "escalation" in memo.lower()

    def test_low_confidence_adds_review_note(self):
        factors = ExtractedFactors()
        memo = compose_audit_memo("Medium", factors, confidence=0.45)
        assert "manual review" in memo.lower()

    def test_high_risk_moderate_confidence_adds_senior_review(self):
        factors = ExtractedFactors(
            weaknesses=["prior loan default"],
        )
        memo = compose_audit_memo("High", factors, confidence=0.68)
        assert "senior credit officer" in memo.lower()

    def test_empty_factors_still_produce_memo(self):
        """Even with no extracted factors, memos should not be empty."""
        for label in ["Low", "Medium", "High"]:
            memo = compose_audit_memo(label, ExtractedFactors(), confidence=0.9)
            assert len(memo) > 20
            assert isinstance(memo, str)

    def test_end_to_end_extraction_and_composition(self):
        """Full pipeline: raw text → factors → memo."""
        text = (
            "Borrower seeking $15M for Senior Housing project in Detroit. "
            "Limited operating history. Strong net worth with diversified "
            "real estate holdings."
        )
        factors = extract_risk_factors(text)
        assert factors.project_type == "Senior Housing"
        assert factors.location == "Detroit"

        memo = compose_audit_memo("Medium", factors, confidence=0.73)
        assert len(memo) > 40
        assert "Detroit" in memo or "Senior Housing" in memo or "$15M" in memo
