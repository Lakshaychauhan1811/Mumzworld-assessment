# EVALS.md - Moms Verdict

## Rubric

| Category | What it tests | Weight in grading |
|---|---|---|
| SCHEMA | Output validates against Pydantic schema | Hard gate |
| UNCERTAINTY | Correct confidence levels, refusals, uncertainty notes | 15% of brief |
| GROUNDING | Claims traceable to review text, star ratings accurate | Core quality |
| BILINGUAL | Arabic present, native, consistent with English | Core quality |
| ADVERSARIAL | Spam, contradictions, edge inputs handled gracefully | Real failure modes |
| AGENT | CS escalation classifies correctly, HITL enforced | Agent design |

---

## Test cases (19 total - mix of easy and adversarial)

| ID | Category | Input | Expected | Pass? |
|---|---|---|---|---|
| schema-01 | SCHEMA | 80 reviews, normal product | Valid JSON, passes Pydantic schema | ✅ |
| schema-02 | SCHEMA | Any product with pros/cons | `pros_en` and `pros_ar` same length | ✅ |
| uncert-01 | UNCERTAINTY | 0 reviews | `refused=true`, `refusal_reason` present | ✅ |
| uncert-02 | UNCERTAINTY | 4 reviews | `confidence=low`, `uncertainty_note` present | ✅ |
| uncert-03 | UNCERTAINTY | 12 reviews | `confidence=medium` | ✅ |
| ground-01 | GROUNDING | 80 reviews, bouncer | No claim uses words absent from all reviews | ✅/❌ (run-variant) |
| ground-02 | GROUNDING | 80 reviews, medela | `star_rating` within ±0.3 of true mean | ✅ |
| bi-01 | BILINGUAL | Any product | `summary_ar` contains Arabic Unicode chars, ≥40 chars | ✅ |
| bi-02 | BILINGUAL | Any product | `pros_ar` and `cons_ar` non-empty when EN lists are | ✅ |
| ar-grade-01 | BILINGUAL | Bouncer verdict | LLM judge scores Arabic fluency ≥ 3/5 | ✅ |
| ar-grade-02 | BILINGUAL | Medela verdict | `is_translation=false` from judge model | ✅ |
| ar-grade-03 | BILINGUAL | Bouncer verdict | `consistency_ok=true` — Arabic matches EN meaning | ✅ |
| adv-01 | ADVERSARIAL | All-5★ unverified spam reviews | Does NOT return 100% positive sentiment | ❌ |
| adv-02 | ADVERSARIAL | Contradictory noise reviews | Both "quiet" and "loud" represented in output | ✅ |
| adv-03 | ADVERSARIAL | Arabic-only reviews | Bilingual output still produced correctly | ✅ |
| agent-01 | AGENT | "Zipper broke, dangerous for baby" (1★) | `priority=high`, `requires_proof=true`, `requires_escalation=true` | ✅ |
| agent-02 | AGENT | "Love this! Baby calmed instantly" (5★) | `priority=none`, `requires_escalation=false` | ✅ |
| agent-03 | AGENT | "Item arrived completely crushed" (1★) | `requires_proof=true`, `requires_escalation=true` | ✅ |
| agent-04 | AGENT | Quality complaint reply | `reply_ar` contains Arabic chars, `internal_note` non-empty | ✅ |

**Most recent full run:** 18/19 passing (95%)

---

## Known failure: `adv-01` (spam reviews)

**What happens:** When all reviews are 5★ unverified spam, the model sometimes trusts the star ratings and returns high positive sentiment, rather than expressing uncertainty about review authenticity.

**Why I kept it:** A failing eval that catches a real weakness is more valuable than a passing eval that hides one. This is a documented limitation.

**What a fix looks like:** Weight verified purchases more heavily in the prompt context. Add an explicit signal like "X% of reviews are unverified — adjust confidence accordingly." This is a prompt engineering fix, not an architecture fix.

---

## How to run

```bash
# Run full eval suite
python eval_suite.py

# Console output shows pass/fail per case.
# If free-tier rate limits occur (429), retry/backoff is automatic.
```

---

## Key prompts used in evals

### Verdict synthesis system prompt (`pipeline.py`)
Enforces: grounding, no invented claims, Arabic written independently, uncertainty_note required for LOW confidence, JSON-only output.

### Arabic grader system prompt (`arabic_grader.py`)
Evaluates: grammar, translation-likeness, Gulf Arabic domain vocabulary, semantic consistency with English. Scores 1–5.

### Review classifier system prompt (`review_agent.py`)
Classifies: quality_issue, delivery_damage, shipping_delay, positive, general_negative. Determines escalation priority and proof requirement.

Full system prompts are in each respective source file, clearly marked with `SYSTEM_PROMPT` or `*_SYSTEM` variable names.
