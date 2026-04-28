"""
Moms Verdict — output schema.

Every field is either populated from review evidence or explicitly null.
The model must never invent a claim not supported by at least one review.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class ConfidenceLevel(str, Enum):
    HIGH = "high"        # 20+ reviews, consistent signal
    MEDIUM = "medium"    # 5–19 reviews, some variance
    LOW = "low"          # < 5 reviews — surface verdict but warn loudly
    INSUFFICIENT = "insufficient"  # 0 reviews — refuse to generate verdict


class SentimentBreakdown(BaseModel):
    positive_pct: float = Field(..., ge=0, le=100, description="% of reviews with positive sentiment")
    negative_pct: float = Field(..., ge=0, le=100, description="% of reviews with negative sentiment")
    neutral_pct:  float = Field(..., ge=0, le=100, description="% of reviews with neutral sentiment")

    @model_validator(mode="after")
    def must_sum_to_100(self) -> "SentimentBreakdown":
        total = self.positive_pct + self.negative_pct + self.neutral_pct
        if not (99.0 <= total <= 101.0):
            raise ValueError(f"Sentiment percentages must sum to 100, got {total:.1f}")
        return self


class Theme(BaseModel):
    label_en: str = Field(..., description="Theme label in English (≤ 5 words)")
    label_ar: str = Field(..., description="Theme label in Arabic (≤ 5 words)")
    sentiment: str = Field(..., pattern="^(positive|negative|mixed)$")
    mention_count: int = Field(..., ge=1, description="Number of reviews mentioning this theme")
    representative_quote: Optional[str] = Field(
        None,
        description="Verbatim short quote from a review supporting this theme. Null if no clean quote available."
    )


class Verdict(BaseModel):
    summary_en: str = Field(
        ...,
        min_length=40,
        max_length=400,
        description="Plain-English verdict paragraph. Grounded in reviews. No invented claims."
    )
    summary_ar: str = Field(
        ...,
        min_length=40,
        max_length=600,  # Arabic is generally wordier
        description="Arabic verdict paragraph. Native copy — not a translation of summary_en."
    )
    star_rating: float = Field(..., ge=1.0, le=5.0, description="Mean star rating from reviews, rounded to 1dp")
    review_count: int = Field(..., ge=0)
    confidence: ConfidenceLevel
    sentiment: SentimentBreakdown
    top_themes: list[Theme] = Field(..., min_length=0, max_length=6)
    pros_en: list[str] = Field(..., min_length=0, max_length=5, description="Top positives in English")
    pros_ar: list[str] = Field(..., min_length=0, max_length=5, description="Top positives in Arabic")
    cons_en: list[str] = Field(..., min_length=0, max_length=5, description="Top negatives in English")
    cons_ar: list[str] = Field(..., min_length=0, max_length=5, description="Top negatives in Arabic")
    would_recommend_pct: Optional[float] = Field(
        None, ge=0, le=100,
        description="% of reviewers who explicitly recommend. Null if not inferable."
    )
    uncertainty_note: Optional[str] = Field(
        None,
        description="Human-readable caveat when confidence is LOW or INSUFFICIENT. Must be present when confidence != HIGH or MEDIUM."
    )

    @model_validator(mode="after")
    def uncertainty_note_required_for_low_confidence(self) -> "Verdict":
        if self.confidence in (ConfidenceLevel.LOW, ConfidenceLevel.INSUFFICIENT):
            if not self.uncertainty_note:
                raise ValueError("uncertainty_note is required when confidence is LOW or INSUFFICIENT")
        return self

    @model_validator(mode="after")
    def pros_cons_balanced(self) -> "Verdict":
        if len(self.pros_en) != len(self.pros_ar):
            raise ValueError("pros_en and pros_ar must have the same number of items")
        if len(self.cons_en) != len(self.cons_ar):
            raise ValueError("cons_en and cons_ar must have the same number of items")
        return self


class MomsVerdictResponse(BaseModel):
    """Top-level response. Either a verdict or a hard refusal."""
    product_name: str
    verdict: Optional[Verdict] = None
    refused: bool = Field(default=False)
    refusal_reason: Optional[str] = Field(
        None,
        description="Why the verdict was refused. Present only when refused=True."
    )

    @model_validator(mode="after")
    def verdict_xor_refusal(self) -> "MomsVerdictResponse":
        if self.refused and self.verdict is not None:
            raise ValueError("Cannot have both a verdict and refused=True")
        if not self.refused and self.verdict is None:
            raise ValueError("Must have either a verdict or refused=True")
        if self.refused and not self.refusal_reason:
            raise ValueError("refusal_reason is required when refused=True")
        return self
