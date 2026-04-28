# Moms Verdict - AI Review Synthesis for Mumzworld

**Track A - AI Engineering Intern | Lakshay Chauhan**

---

## Summary

Moms Verdict synthesizes product reviews into a structured bilingual verdict in English and Arabic to help mothers on Mumzworld make faster, safer buying decisions. It includes strict schema validation, explicit uncertainty handling (including hard refusal on zero evidence), an Arabic fluency judge (LLM-as-judge), and a review-reply agent with human-in-the-loop escalation for sensitive CS cases.

---

## Prototype access

- **GitHub repo**: https://github.com/Lakshaychauhan1811/Mumzworld-assessment
- **Loom walkthrough (3 min)**: https://www.loom.com/share/53189065a414416dbef83962554892fa

---

## Setup and run (under 5 minutes)

### 1) Install

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

### 2) API key

Create a `.env` file in project root:

```bash
GROQ_API_KEY=gsk_...
```

(`pipeline.py` auto-loads `.env`, so no manual export is required.)

### 3) Run end to end

```bash
# 1) Generate synthetic data (no API call)
python generate_reviews.py

# 2) Run verdict pipeline on all products
python pipeline.py --all

# 3) Run review-reply agent (HITL escalation)
python review_agent.py --review "The zipper broke after two days, dangerous for my baby"

# 4) Run eval suite
python eval_suite.py
```

---

## Project structure

```text
.
├── schema.py
├── pipeline.py
├── arabic_grader.py
├── review_agent.py
├── generate_reviews.py
├── eval_suite.py
├── data/            # generated artifacts
├── escalations/     # escalation queue for CS review
├── EVALS.md
├── TRADEOFFS.md
└── README.md
```

---

## Evals

See **[EVALS.md](EVALS.md)** for rubric, 19 test cases, scores, and failure analysis.

Current summary: **18/19 passing (95%)**.  
Known fail: `adv-01` (all-spam unverified reviews can still look over-positive).

---

## Tradeoffs

See **[TRADEOFFS.md](TRADEOFFS.md)** for architecture choices, uncertainty strategy, rejected options, and next steps.

---

## Tooling

- **Cursor + AI pair-coding workflow**: used for implementation speed, refactoring, and debugging loops.
- **Groq API (`llama-3.3-70b-versatile`)**: primary model for verdict synthesis, classification, and bilingual reply generation.
- **Pydantic**: hard schema gate to reject malformed or unsafe outputs.
- **What worked**: structured JSON prompting + validators + eval-first iteration.
- **What did not**: free-tier 429 rate limits; mitigated with retry/backoff and explicit reporting in demo/evals.

---

## AI usage note (max 5 lines)

- Used AI-assisted coding for scaffolding, prompt iteration, eval design, and debugging.
- Core product decisions (uncertainty policy, refusal behavior, HITL safety gates, eval criteria) were human-authored.
- LLM inference runs via Groq; code supports OpenRouter/Anthropic fallback.
- Prompts that materially shape behavior are committed in `pipeline.py`, `review_agent.py`, and `arabic_grader.py`.
- This submission documents failures and limits explicitly rather than hiding them.

---

## Time log (max 5 lines)

- Problem scoping + schema design: ~50 min
- Data generation + pipeline implementation: ~120 min
- Validation + edge cases + retry/backoff hardening: ~60 min
- Eval suite + failure analysis: ~60 min
- Docs + Loom prep: ~50 min (total ~5h 40m; overrun due to adversarial eval and rate-limit handling)
