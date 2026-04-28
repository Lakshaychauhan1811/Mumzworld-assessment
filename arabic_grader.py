"""
Arabic Fluency Self-Grader

Uses a second LLM call to evaluate whether the Arabic output is:
  1. Grammatically correct (not gibberish or transliteration)
  2. Native-sounding (not a literal word-for-word translation of the English)
  3. Contextually appropriate (baby/parenting domain vocabulary)
  4. Consistent with the English verdict (same meaning, not contradictory)

This is a deliberate second LLM call — a separate judge model grades the
output of the synthesis model. This is a standard eval pattern (LLM-as-judge)
and directly addresses the known weakness: we can't manually verify Arabic.

Returns: ArabicGradeResult with a score 1–5, pass/fail, and specific feedback.

Usage:
  from src.arabic_grader import grade_arabic
  result = grade_arabic(summary_en, summary_ar, pros_en, pros_ar, cons_en, cons_ar)
"""

import json
import sys
from pathlib import Path
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import call_llm


# ── Output schema for the grader ─────────────────────────────────────────────

class ArabicGradeResult(BaseModel):
    score: int = Field(..., ge=1, le=5, description="Overall fluency score 1–5")
    pass_threshold: bool = Field(..., description="True if score >= 3 (minimum acceptable)")
    is_translation: bool = Field(..., description="True if Arabic reads like a direct translation of English")
    grammar_ok: bool = Field(..., description="True if no obvious grammatical errors detected")
    domain_vocab_ok: bool = Field(..., description="True if baby/parenting vocabulary is appropriate")
    consistency_ok: bool = Field(..., description="True if Arabic meaning matches English meaning")
    feedback_en: str = Field(..., description="Specific feedback in English about the Arabic quality")
    worst_phrase: str | None = Field(None, description="Most problematic phrase in the Arabic, if any")


# ── Grader prompt ─────────────────────────────────────────────────────────────

GRADER_SYSTEM = """You are an expert Arabic language evaluator specialising in e-commerce copy for Gulf Arabic markets (UAE, Saudi Arabia, Kuwait).

Your job: evaluate whether Arabic product review summaries are genuinely native-quality, not translated English.

You will receive:
- An English verdict summary and pros/cons
- The corresponding Arabic verdict summary and pros/cons

Score the Arabic on a scale of 1–5:
  5 = Reads like it was written by a native Arabic copywriter. Natural idioms, correct MSA or Gulf dialect, appropriate register.
  4 = Very good. Minor awkwardness but clearly not a translation. A native would accept it.
  3 = Acceptable. Some translated phrasing but the meaning is clear and grammar is mostly correct.
  2 = Poor. Obvious word-for-word translation. Awkward sentence structure. A native would notice immediately.
  1 = Unacceptable. Garbled, wrong grammar, transliteration of English words, or pure gibberish.

Pass threshold: score >= 3.

Key things to check:
- Does it use natural Arabic sentence structure (VSO order where appropriate)?
- Does it avoid direct calques from English idioms? (e.g. "يستحق كل قرش" is good; "يستحق المال" is a weak translation of "worth the money")
- Does it use appropriate parenting/baby vocabulary used in Gulf Arabic? (حفاضات، رضاعة، مهد، etc.)
- Is the register consistent? (Formal MSA or natural Gulf informal — not mixed awkwardly)
- Does the Arabic meaning match the English meaning? (Consistency check)

Respond with ONLY valid JSON, no preamble:
{
  "score": integer 1-5,
  "pass_threshold": boolean,
  "is_translation": boolean,
  "grammar_ok": boolean,
  "domain_vocab_ok": boolean,
  "consistency_ok": boolean,
  "feedback_en": "specific feedback string",
  "worst_phrase": "most problematic Arabic phrase or null"
}"""


def grade_arabic(
    summary_en: str,
    summary_ar: str,
    pros_en: list[str],
    pros_ar: list[str],
    cons_en: list[str],
    cons_ar: list[str],
    product_name: str = "",
) -> ArabicGradeResult:
    """
    Run a second LLM call to grade Arabic fluency.
    Uses the same call_llm() so it works with Groq/OpenRouter/Anthropic.
    """
    user_prompt = f"""Product: {product_name}

=== ENGLISH ===
Summary: {summary_en}
Pros: {pros_en}
Cons: {cons_en}

=== ARABIC (to evaluate) ===
Summary: {summary_ar}
Pros: {pros_ar}
Cons: {cons_ar}

Evaluate the Arabic quality. Return only JSON."""

    raw = call_llm(GRADER_SYSTEM, user_prompt)

    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    data = json.loads(cleaned)
    return ArabicGradeResult.model_validate(data)


def grade_verdict(verdict_dict: dict) -> ArabicGradeResult | None:
    """Convenience wrapper: grade from a pipeline verdict dict."""
    v = verdict_dict.get("verdict")
    if not v:
        return None  # Refused — nothing to grade
    return grade_arabic(
        summary_en=v["summary_en"],
        summary_ar=v["summary_ar"],
        pros_en=v.get("pros_en", []),
        pros_ar=v.get("pros_ar", []),
        cons_en=v.get("cons_en", []),
        cons_ar=v.get("cons_ar", []),
        product_name=verdict_dict.get("product_name", ""),
    )


# ── CLI usage ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grade Arabic fluency of a verdict")
    parser.add_argument("--verdicts", default="data/verdicts.json",
                        help="Path to verdicts.json produced by pipeline.py --all")
    parser.add_argument("--product", help="Grade a specific product ID only")
    args = parser.parse_args()

    verdicts_path = Path(args.verdicts)
    if not verdicts_path.exists():
        print("Run pipeline.py --all first to generate verdicts.json")
        sys.exit(1)

    with open(verdicts_path, encoding="utf-8") as f:
        all_verdicts = json.load(f)

    product_ids = [args.product] if args.product else list(all_verdicts.keys())

    print(f"\n{'═'*60}")
    print(f"  Arabic Fluency Grader")
    print(f"{'═'*60}")

    results = {}
    for pid in product_ids:
        verdict = all_verdicts.get(pid)
        if not verdict:
            print(f"\n  [{pid}] Not found in verdicts.json")
            continue

        print(f"\n  [{pid}] Grading Arabic…", end=" ", flush=True)
        try:
            grade = grade_verdict(verdict)
            if grade is None:
                print("SKIPPED (refused verdict)")
                continue

            status = "PASS" if grade.pass_threshold else "FAIL"
            print(f"{status}  (score: {grade.score}/5)")
            print(f"    Grammar OK     : {grade.grammar_ok}")
            print(f"    Not translation: {not grade.is_translation}")
            print(f"    Domain vocab   : {grade.domain_vocab_ok}")
            print(f"    Consistent     : {grade.consistency_ok}")
            print(f"    Feedback       : {grade.feedback_en}")
            if grade.worst_phrase:
                print(f"    Worst phrase   : {grade.worst_phrase}")

            results[pid] = grade.model_dump()
        except Exception as e:
            print(f"ERROR: {e}")
            results[pid] = {"error": str(e)}

    out_path = Path("evals/arabic_grades.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n  Results → {out_path}")
    print(f"{'═'*60}\n")
