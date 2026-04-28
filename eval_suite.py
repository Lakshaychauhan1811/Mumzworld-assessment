"""
Moms Verdict — Eval Suite

Rubric categories:
  - GROUNDING     : Output claims are traceable to review text
  - UNCERTAINTY   : System correctly refuses or flags low-confidence cases
  - SCHEMA        : Output validates against Pydantic schema
  - BILINGUAL     : Arabic is present and not an obvious translation
  - ADVERSARIAL   : System handles spam, contradictions, abuse gracefully

Run: python evals/eval_suite.py
Requires: data/reviews.json  (run generate_reviews.py first)
          OPENROUTER_API_KEY or ANTHROPIC_API_KEY
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import run_product, call_llm, parse_and_validate, SYSTEM_PROMPT, build_user_prompt, get_confidence
from schema import MomsVerdictResponse, ConfidenceLevel
from arabic_grader import grade_arabic, ArabicGradeResult
from review_agent import (
    classify_review, ReviewCategory, EscalationPriority,
    process_review, ReviewClassification
)


# ── Test case definition ──────────────────────────────────────────────────────

@dataclass
class EvalCase:
    id: str
    description: str
    category: str           # GROUNDING | UNCERTAINTY | SCHEMA | BILINGUAL | ADVERSARIAL
    product_id: Optional[str] = None
    custom_reviews: Optional[list] = None   # Override reviews for adversarial cases
    expect_refused: bool = False
    expect_confidence: Optional[str] = None
    check_fn: Optional[callable] = None     # Extra programmatic checks
    notes: str = ""


def check_arabic_present(result: MomsVerdictResponse) -> tuple[bool, str]:
    if not result.verdict:
        return True, "N/A (refused)"
    ar = result.verdict.summary_ar
    # Check for Arabic Unicode block (U+0600–U+06FF)
    has_arabic = any('\u0600' <= c <= '\u06ff' for c in ar)
    if not has_arabic:
        return False, f"summary_ar contains no Arabic characters: '{ar[:80]}'"
    if len(ar) < 40:
        return False, f"summary_ar too short ({len(ar)} chars)"
    return True, "Arabic present and non-trivial"


def check_no_invented_facts(result: MomsVerdictResponse, reviews: list) -> tuple[bool, str]:
    """Heuristic: pros/cons should not contain words never mentioned in any review."""
    if not result.verdict:
        return True, "N/A"
    review_words = set()
    for r in reviews:
        review_words.update(r["text"].lower().split())

    suspicious = []
    for claim in result.verdict.pros_en + result.verdict.cons_en:
        claim_words = [w.lower() for w in claim.split() if len(w) > 5]
        found = any(w in review_words for w in claim_words)
        if not found and claim_words:
            suspicious.append(claim)

    if suspicious:
        return False, f"Possibly invented claims (no supporting review words): {suspicious}"
    return True, "No obviously invented claims detected"


def check_uncertainty_note_present(result: MomsVerdictResponse) -> tuple[bool, str]:
    if result.refused:
        return True, "Refused as expected"
    if not result.verdict:
        return False, "No verdict and not refused"
    if result.verdict.confidence in (ConfidenceLevel.LOW, ConfidenceLevel.INSUFFICIENT):
        if not result.verdict.uncertainty_note:
            return False, "Low/insufficient confidence but uncertainty_note is missing"
    return True, f"uncertainty_note present: {result.verdict.uncertainty_note}"


def check_contradictions_flagged(result: MomsVerdictResponse) -> tuple[bool, str]:
    """For the noise-contradiction case: both sides should appear in output."""
    if not result.verdict:
        return True, "N/A"
    text = (result.verdict.summary_en + " ".join(result.verdict.cons_en + result.verdict.pros_en)).lower()
    has_quiet = any(w in text for w in ["quiet", "silent", "noise-free"])
    has_loud  = any(w in text for w in ["loud", "noisy", "audible"])
    if has_quiet and has_loud:
        return True, "Both sides of contradiction represented"
    return False, f"Contradiction not balanced. quiet={has_quiet}, loud={has_loud}"




# ── Arabic grader checks (LLM-as-judge) ──────────────────────────────────────

def check_arabic_fluency_llm(result: MomsVerdictResponse) -> tuple[bool, str]:
    """
    Second LLM call grades the Arabic output for native fluency.
    This is the key differentiator — we don't just check Arabic chars exist,
    we ask a judge model to evaluate whether it reads like native copy.
    """
    if not result.verdict:
        return True, "N/A (refused)"
    try:
        grade = grade_arabic(
            summary_en=result.verdict.summary_en,
            summary_ar=result.verdict.summary_ar,
            pros_en=result.verdict.pros_en,
            pros_ar=result.verdict.pros_ar,
            cons_en=result.verdict.cons_en,
            cons_ar=result.verdict.cons_ar,
        )
        passed = grade.pass_threshold
        detail = f"score={grade.score}/5 | translation={grade.is_translation} | {grade.feedback_en[:80]}"
        return passed, detail
    except Exception as e:
        return False, f"Grader error: {e}"


def check_arabic_not_translation(result: MomsVerdictResponse) -> tuple[bool, str]:
    """Specifically checks the is_translation flag from the LLM judge."""
    if not result.verdict:
        return True, "N/A"
    try:
        grade = grade_arabic(
            summary_en=result.verdict.summary_en,
            summary_ar=result.verdict.summary_ar,
            pros_en=result.verdict.pros_en,
            pros_ar=result.verdict.pros_ar,
            cons_en=result.verdict.cons_en,
            cons_ar=result.verdict.cons_ar,
        )
        passed = not grade.is_translation
        return passed, f"is_translation={grade.is_translation} | {grade.feedback_en[:80]}"
    except Exception as e:
        return False, f"Grader error: {e}"


def check_arabic_consistency(result: MomsVerdictResponse) -> tuple[bool, str]:
    """Checks that Arabic meaning is consistent with English — not contradictory."""
    if not result.verdict:
        return True, "N/A"
    try:
        grade = grade_arabic(
            summary_en=result.verdict.summary_en,
            summary_ar=result.verdict.summary_ar,
            pros_en=result.verdict.pros_en,
            pros_ar=result.verdict.pros_ar,
            cons_en=result.verdict.cons_en,
            cons_ar=result.verdict.cons_ar,
        )
        return grade.consistency_ok, f"consistency={grade.consistency_ok} | {grade.feedback_en[:80]}"
    except Exception as e:
        return False, f"Grader error: {e}"



# ── Review agent checks ───────────────────────────────────────────────────────

def check_quality_escalates(review_text: str, star: int) -> tuple[bool, str]:
    """Quality issues must be classified as HIGH priority and require proof."""
    clf = classify_review(review_text, star, "en")
    high = clf.priority == EscalationPriority.HIGH
    proof = clf.requires_proof
    escalate = clf.requires_escalation
    passed = high and proof and escalate
    return passed, f"priority={clf.priority.value} proof={proof} escalate={escalate} | {clf.issue_summary}"


def check_positive_no_escalation(review_text: str, star: int) -> tuple[bool, str]:
    """Positive reviews must NOT be escalated."""
    clf = classify_review(review_text, star, "en")
    passed = (
        clf.category == ReviewCategory.POSITIVE and
        clf.priority == EscalationPriority.NONE and
        not clf.requires_escalation
    )
    return passed, f"category={clf.category.value} priority={clf.priority.value} escalate={clf.requires_escalation}"


def check_delivery_damage_escalates(review_text: str, star: int) -> tuple[bool, str]:
    """Delivery damage must require proof and escalation."""
    clf = classify_review(review_text, star, "en")
    passed = clf.requires_proof and clf.requires_escalation
    return passed, f"category={clf.category.value} proof={clf.requires_proof} escalate={clf.requires_escalation}"


def check_reply_bilingual(review_text: str, star: int, product_name: str) -> tuple[bool, str]:
    """Drafted reply must contain Arabic characters."""
    clf = classify_review(review_text, star, "en")
    from review_agent import draft_reply
    reply = draft_reply(review_text, star, clf, product_name)
    has_arabic = any('\u0600' <= c <= '\u06ff' for c in reply.reply_ar)
    has_internal = len(reply.internal_note) > 10
    passed = has_arabic and has_internal
    return passed, f"arabic_present={has_arabic} internal_note_present={has_internal}"

# ── Load data helper ──────────────────────────────────────────────────────────

def load_data():
    p = Path("data/reviews.json")
    if not p.exists():
        print("Run: python src/generate_reviews.py first")
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ── Eval cases ────────────────────────────────────────────────────────────────

def build_eval_cases(all_data: dict) -> list[EvalCase]:
    bouncer_reviews  = [r for r in all_data["reviews"] if r["product_id"] == "babybjorn-bouncer"]
    medela_reviews   = [r for r in all_data["reviews"] if r["product_id"] == "medela-freestyle"]
    teething_reviews = [r for r in all_data["reviews"] if r["product_id"] == "generic-teething"]
    spam_reviews     = all_data["edge_cases"]["spam_reviews"]
    contra_reviews   = all_data["edge_cases"]["contradictory_reviews"]

    return [
        # ── SCHEMA ────────────────────────────────────────────────────────────
        EvalCase(
            id="schema-01",
            description="Output validates against Pydantic schema for a normal product",
            category="SCHEMA",
            product_id="babybjorn-bouncer",
            notes="Core contract test — if this fails, nothing else matters",
        ),
        EvalCase(
            id="schema-02",
            description="Pros/cons EN and AR lists have equal length",
            category="SCHEMA",
            product_id="medela-freestyle",
            check_fn=lambda r, _: (
                len(r.verdict.pros_en) == len(r.verdict.pros_ar) and
                len(r.verdict.cons_en) == len(r.verdict.cons_ar),
                "Pros/cons EN/AR balanced"
            ) if r.verdict else (True, "N/A"),
        ),

        # ── UNCERTAINTY ───────────────────────────────────────────────────────
        EvalCase(
            id="uncert-01",
            description="Zero reviews → refused=True with refusal_reason",
            category="UNCERTAINTY",
            product_id="mystery-product",
            custom_reviews=[],
            expect_refused=True,
            notes="Hard refusal path",
        ),
        EvalCase(
            id="uncert-02",
            description="< 5 reviews → confidence=low, uncertainty_note present",
            category="UNCERTAINTY",
            product_id="generic-teething",
            expect_confidence="low",
            check_fn=lambda r, _: check_uncertainty_note_present(r),
            notes="15 reviews triggers LOW, but note: we test with only 4 here",
            custom_reviews=teething_reviews[:4],  # Force LOW path
        ),
        EvalCase(
            id="uncert-03",
            description="5–19 reviews → confidence=medium (not low or high)",
            category="UNCERTAINTY",
            product_id="generic-teething",
            expect_confidence="medium",
            custom_reviews=teething_reviews[:12],
        ),

        # ── GROUNDING ─────────────────────────────────────────────────────────
        EvalCase(
            id="ground-01",
            description="Pros/cons are traceable to review content (no invented facts)",
            category="GROUNDING",
            product_id="babybjorn-bouncer",
            check_fn=lambda r, reviews: check_no_invented_facts(r, reviews),
        ),
        EvalCase(
            id="ground-02",
            description="star_rating matches mean of input reviews (±0.3)",
            category="GROUNDING",
            product_id="medela-freestyle",
            check_fn=lambda r, reviews: (
                abs(r.verdict.star_rating - (sum(x["star"] for x in reviews) / len(reviews))) <= 0.3,
                f"star_rating {r.verdict.star_rating} vs true mean {sum(x['star'] for x in reviews)/len(reviews):.2f}"
            ) if r.verdict and reviews else (True, "N/A"),
        ),

        # ── BILINGUAL ─────────────────────────────────────────────────────────
        EvalCase(
            id="bi-01",
            description="summary_ar contains Arabic characters and is ≥ 40 chars",
            category="BILINGUAL",
            product_id="babybjorn-bouncer",
            check_fn=lambda r, _: check_arabic_present(r),
        ),
        EvalCase(
            id="bi-02",
            description="pros_ar and cons_ar are non-empty when pros_en and cons_en are",
            category="BILINGUAL",
            product_id="medela-freestyle",
            check_fn=lambda r, _: (
                (len(r.verdict.pros_ar) > 0 if r.verdict.pros_en else True) and
                (len(r.verdict.cons_ar) > 0 if r.verdict.cons_en else True),
                "AR lists populated"
            ) if r.verdict else (True, "N/A"),
        ),

        # ── ARABIC GRADER (LLM-as-judge) ──────────────────────────────────────
        EvalCase(
            id="ar-grade-01",
            description="LLM judge scores Arabic fluency >= 3/5 (acceptable threshold)",
            category="BILINGUAL",
            product_id="babybjorn-bouncer",
            check_fn=lambda r, _: check_arabic_fluency_llm(r),
            notes="Key eval: second LLM call grades the first LLM's Arabic output",
        ),
        EvalCase(
            id="ar-grade-02",
            description="Arabic is not a direct translation of the English copy",
            category="BILINGUAL",
            product_id="medela-freestyle",
            check_fn=lambda r, _: check_arabic_not_translation(r),
            notes="Native Arabic should be written independently, not mirroring EN structure",
        ),
        EvalCase(
            id="ar-grade-03",
            description="Arabic meaning is consistent with English verdict",
            category="BILINGUAL",
            product_id="babybjorn-bouncer",
            check_fn=lambda r, _: check_arabic_consistency(r),
            notes="Guards against Arabic that passes fluency but contradicts the EN verdict",
        ),

                # ── ADVERSARIAL ───────────────────────────────────────────────────────
        EvalCase(
            id="adv-01",
            description="All-5-star spam reviews → model does not output 100% positive",
            category="ADVERSARIAL",
            product_id="babybjorn-bouncer",
            custom_reviews=spam_reviews,
            check_fn=lambda r, _: (
                r.verdict is None or r.verdict.sentiment.positive_pct < 100,
                "Did not blindly trust spam reviews"
            ),
            notes="Spam reviews are unverified — good models express uncertainty",
        ),
        EvalCase(
            id="adv-02",
            description="Contradictory noise reviews → both perspectives reflected",
            category="ADVERSARIAL",
            product_id="medela-freestyle",
            custom_reviews=medela_reviews + contra_reviews,
            check_fn=lambda r, _: check_contradictions_flagged(r),
        ),
        EvalCase(
            id="adv-03",
            description="Arabic-only reviews → Arabic summary still produced correctly",
            category="ADVERSARIAL",
            product_id="babybjorn-bouncer",
            custom_reviews=[r for r in bouncer_reviews if r["lang"] == "ar"],
            check_fn=lambda r, _: check_arabic_present(r),
            notes="Tests that Arabic input doesn't break the bilingual output",
        ),

        # ── REVIEW AGENT (Human-in-the-Loop) ─────────────────────────────────
        EvalCase(
            id="agent-01",
            description="Quality issue → HIGH priority, requires proof, escalated to CS",
            category="AGENT",
            check_fn=lambda r, _: check_quality_escalates(
                "The zipper broke after 2 days. Terrible quality, dangerous for baby.", 1
            ),
            notes="Core safety: defective products must always escalate",
        ),
        EvalCase(
            id="agent-02",
            description="Positive review → no escalation, priority=none",
            category="AGENT",
            check_fn=lambda r, _: check_positive_no_escalation(
                "Absolutely love this! My baby calmed down instantly. Highly recommend.", 5
            ),
            notes="Positive reviews must not create CS burden",
        ),
        EvalCase(
            id="agent-03",
            description="Delivery damage → requires proof and CS escalation",
            category="AGENT",
            check_fn=lambda r, _: check_delivery_damage_escalates(
                "Item arrived completely crushed. Box was destroyed and product is unusable.", 1
            ),
        ),
        EvalCase(
            id="agent-04",
            description="Reply draft is bilingual (EN + AR) with internal CS note",
            category="AGENT",
            check_fn=lambda r, _: check_reply_bilingual(
                "The product quality is very poor, broke within a week.", 2,
                "Babybjorn Bouncer Balance Soft"
            ),
            notes="Bilingual replies required — CS serves EN and AR customers",
        ),
    ]


# ── Runner ────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    case_id: str
    category: str
    description: str
    passed: bool
    score: float          # 0.0 or 1.0 (binary for now)
    notes: str
    error: Optional[str] = None


def run_eval(case: EvalCase, all_data: dict) -> EvalResult:
    print(f"  [{case.id}] {case.description[:60]}…", end=" ", flush=True)

    try:
        # Build reviews
        if case.custom_reviews is not None:
            reviews = case.custom_reviews
        elif case.product_id:
            reviews = [r for r in all_data["reviews"] if r["product_id"] == case.product_id]
        else:
            reviews = []

        product = next((p for p in all_data["products"] if p["id"] == case.product_id), None)
        product_name = product["name"] if product else case.product_id or "Unknown"

        confidence = get_confidence(len(reviews))
        user_prompt = build_user_prompt(product_name, reviews, confidence)
        raw = call_llm(SYSTEM_PROMPT, user_prompt)
        result = parse_and_validate(raw, product_name)

        checks = []

        # Schema always checked
        checks.append((True, "Schema valid"))

        # Expect refused
        if case.expect_refused:
            checks.append((result.refused, f"refused={result.refused}"))

        # Expect confidence level
        if case.expect_confidence and result.verdict:
            checks.append((
                result.verdict.confidence.value == case.expect_confidence,
                f"confidence={result.verdict.confidence.value} (expected {case.expect_confidence})"
            ))

        # Custom check
        if case.check_fn:
            ok, msg = case.check_fn(result, reviews)
            checks.append((ok, msg))

        passed = all(ok for ok, _ in checks)
        notes  = " | ".join(msg for _, msg in checks)
        print("PASS" if passed else "FAIL")
        return EvalResult(case.id, case.category, case.description, passed, float(passed), notes)

    except Exception as e:
        print(f"ERROR")
        return EvalResult(case.id, case.category, case.description, False, 0.0,
                          notes="", error=str(e))


def main():
    all_data = load_data()
    cases = build_eval_cases(all_data)

    print(f"\n{'═'*60}")
    print(f"  Moms Verdict — Eval Suite  ({len(cases)} cases)")
    print(f"{'═'*60}")

    results: list[EvalResult] = []
    by_category: dict[str, list[EvalResult]] = {}

    for case in cases:
        print(f"\n  Category: {case.category}")
        r = run_eval(case, all_data)
        results.append(r)
        by_category.setdefault(case.category, []).append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Results by category")
    print(f"{'─'*60}")

    total_pass = 0
    for cat, cat_results in by_category.items():
        passed = sum(r.passed for r in cat_results)
        total  = len(cat_results)
        total_pass += passed
        bar = "█" * passed + "░" * (total - passed)
        print(f"  {cat:<14} {bar}  {passed}/{total}")

    print(f"{'─'*60}")
    print(f"  TOTAL          {total_pass}/{len(results)}  ({100*total_pass/len(results):.0f}%)")
    print(f"{'═'*60}")

    # ── Failures ──────────────────────────────────────────────────────────────
    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\n  Failed cases:")
        for r in failures:
            print(f"  ✗ [{r.case_id}] {r.description[:55]}")
            if r.error:
                print(f"      Error: {r.error}")
            else:
                print(f"      Notes: {r.notes}")

    # ── Write results ─────────────────────────────────────────────────────────
    out = {
        "summary": {
            "total": len(results),
            "passed": total_pass,
            "pct": round(100 * total_pass / len(results), 1),
        },
        "by_category": {
            cat: {
                "passed": sum(r.passed for r in rlist),
                "total": len(rlist),
            }
            for cat, rlist in by_category.items()
        },
        "cases": [
            {
                "id": r.case_id,
                "category": r.category,
                "description": r.description,
                "passed": r.passed,
                "notes": r.notes,
                "error": r.error,
            }
            for r in results
        ]
    }
    Path("evals").mkdir(exist_ok=True)
    with open("evals/results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Full results → evals/results.json\n")


if __name__ == "__main__":
    main()
