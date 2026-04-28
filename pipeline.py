"""
Moms Verdict — synthesis pipeline.

Usage:
  python src/pipeline.py --product babybjorn-bouncer
  python src/pipeline.py --product medela-freestyle
  python src/pipeline.py --product generic-teething     # LOW confidence
  python src/pipeline.py --product mystery-product      # INSUFFICIENT — should refuse
  python src/pipeline.py --all                          # run all products

Environment:
  OPENROUTER_API_KEY  — free at openrouter.ai (or set ANTHROPIC_API_KEY for direct)
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests
from pydantic import ValidationError

# Ensure Unicode logs render on Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import MomsVerdictResponse, ConfidenceLevel

# ── Config ────────────────────────────────────────────────────────────────────

def _load_dotenv_if_present() -> None:
    """Load a local .env file into process env vars (non-destructive)."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv_if_present()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")

# Model selection
# Groq free models: llama-3.3-70b-versatile, llama3-8b-8192, mixtral-8x7b-32768
DEFAULT_MODEL = "llama-3.3-70b-versatile"

MAX_REVIEWS_IN_PROMPT = 60   # Truncate to avoid context limits; prioritize verified

# ── Confidence thresholds ─────────────────────────────────────────────────────

def get_confidence(count: int) -> ConfidenceLevel:
    if count == 0:
        return ConfidenceLevel.INSUFFICIENT
    elif count < 5:
        return ConfidenceLevel.LOW
    elif count < 20:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.HIGH


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Moms Verdict engine for Mumzworld, the largest baby e-commerce platform in the Middle East.

Your job: synthesize product reviews into a structured bilingual verdict that helps mothers make confident purchase decisions.

STRICT RULES — violations disqualify the output:
1. Every claim must be traceable to at least one review. If a claim is not in the reviews, do not make it.
2. Arabic copy must read as native Arabic — NOT a translation of the English text. Write it independently.
3. If review_count < 5, set confidence to "low" and include a clear uncertainty_note.
4. If review_count == 0, set refused=true and explain why.
5. Never pad output with generic marketing language ("great for all babies", "perfect gift", etc.)
6. If reviews contradict each other on a topic (e.g. noise level), acknowledge both sides — do not pick one.
7. Represent minority opinions if they appear in ≥ 10% of reviews.

OUTPUT FORMAT: Respond with ONLY valid JSON matching this exact schema — no preamble, no markdown fences:

{
  "product_name": "string",
  "refused": false,
  "refusal_reason": null,
  "verdict": {
    "summary_en": "string (40–400 chars)",
    "summary_ar": "string (40–600 chars, native Arabic)",
    "star_rating": number,
    "review_count": integer,
    "confidence": "high|medium|low|insufficient",
    "sentiment": {
      "positive_pct": number,
      "negative_pct": number,
      "neutral_pct": number
    },
    "top_themes": [
      {
        "label_en": "string (≤5 words)",
        "label_ar": "string (≤5 words)",
        "sentiment": "positive|negative|mixed",
        "mention_count": integer,
        "representative_quote": "string or null"
      }
    ],
    "pros_en": ["string"],
    "pros_ar": ["string"],
    "cons_en": ["string"],
    "cons_ar": ["string"],
    "would_recommend_pct": number or null,
    "uncertainty_note": "string or null"
  }
}

When refused=true: set verdict to null, refused to true, refusal_reason to a clear explanation.
"""

def build_user_prompt(product_name: str, reviews: list[dict], confidence: ConfidenceLevel) -> str:
    review_count = len(reviews)

    if review_count == 0:
        return f"""Product: {product_name}
Reviews: None available.
review_count: 0
confidence: insufficient

Generate a refusal. Do not invent any verdict."""

    # Compute star rating
    stars = [r["star"] for r in reviews]
    mean_star = round(statistics.mean(stars), 1)

    # Format reviews compactly
    review_lines = []
    for r in reviews[:MAX_REVIEWS_IN_PROMPT]:
        flag = "✓" if r.get("verified_purchase") else "?"
        review_lines.append(f"[{r['lang'].upper()} {r['star']}★ {flag}] {r['text']}")

    reviews_block = "\n".join(review_lines)

    confidence_instruction = ""
    if confidence == ConfidenceLevel.LOW:
        confidence_instruction = "\nWARNING: Only {count} reviews available. Set confidence=low and include a clear uncertainty_note warning the reader that this verdict is based on limited data.".format(count=review_count)

    return f"""Product: {product_name}
Total reviews: {review_count}
Mean star rating: {mean_star}
Confidence level: {confidence.value}
{confidence_instruction}

Reviews (format: [LANG STARS verified?] text):
{reviews_block}

Synthesize a Moms Verdict. Remember: Arabic must be independently written native copy, not a translation."""


# ── API call ─────────────────────────────────────────────────────────────────

def call_llm(system: str, user: str, model: str = DEFAULT_MODEL) -> str:
    """Call Groq, OpenRouter, or Anthropic — whichever key is set. Groq checked first."""

    if GROQ_API_KEY:
        # Groq uses OpenAI-compatible endpoint — fast, generous free tier.
        # Retry on rate limits/transient failures to avoid crashing a full --all run.
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000,
                },
                timeout=60,
            )

            if response.status_code < 400:
                return response.json()["choices"][0]["message"]["content"]

            if response.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts:
                retry_after = response.headers.get("Retry-After")
                wait_s = float(retry_after) if retry_after else min(2 ** attempt, 20)
                print(f"Groq temporary error {response.status_code}. Retrying in {wait_s:.1f}s (attempt {attempt}/{max_attempts})...")
                time.sleep(wait_s)
                continue

            response.raise_for_status()

    elif OPENROUTER_API_KEY:
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://mumzworld-intern-assessment.local",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000,
                },
                timeout=60,
            )
            if response.status_code < 400:
                return response.json()["choices"][0]["message"]["content"]
            if response.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts:
                time.sleep(min(2 ** attempt, 20))
                continue
            response.raise_for_status()

    elif ANTHROPIC_API_KEY:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 2000,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]

    else:
        raise EnvironmentError(
            "No API key found. Set one of:\n"
            "  GROQ_API_KEY       - free at console.groq.com (recommended)\n"
            "  OPENROUTER_API_KEY - free at openrouter.ai\n"
            "  ANTHROPIC_API_KEY  - paid\n"
        )
# ── Parse + validate ──────────────────────────────────────────────────────────

def parse_and_validate(raw: str, product_name: str) -> MomsVerdictResponse:
    """Strip any accidental markdown fences, parse JSON, validate schema."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    data = json.loads(cleaned)
    return MomsVerdictResponse.model_validate(data)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_product(product_id: str, all_data: dict, model: str = DEFAULT_MODEL) -> dict:
    product = next((p for p in all_data["products"] if p["id"] == product_id), None)
    if not product:
        raise ValueError(f"Product '{product_id}' not found in reviews.json")

    reviews = [r for r in all_data["reviews"] if r["product_id"] == product_id]

    # Prioritise verified purchases, then shuffle for variety
    verified   = [r for r in reviews if r.get("verified_purchase")]
    unverified = [r for r in reviews if not r.get("verified_purchase")]
    reviews_for_prompt = (verified + unverified)[:MAX_REVIEWS_IN_PROMPT]

    confidence = get_confidence(len(reviews))
    print(f"\n{'─'*60}")
    print(f"Product : {product['name']}")
    print(f"Reviews : {len(reviews)}  |  Confidence: {confidence.value}")

    if len(reviews) == 0:
        # Hard refusal path should not burn an API call.
        refusal = MomsVerdictResponse(
            product_name=product["name"],
            verdict=None,
            refused=True,
            refusal_reason="No reviews available for this product, so a grounded verdict cannot be generated."
        )
        print("Status  : REFUSED")
        print(f"Refusal : {refusal.refusal_reason}")
        return refusal.model_dump()

    user_prompt = build_user_prompt(product["name"], reviews_for_prompt, confidence)

    print("Calling LLM…")
    raw = call_llm(SYSTEM_PROMPT, user_prompt, model)

    print("Validating schema…")
    try:
        result = parse_and_validate(raw, product["name"])
        status = "REFUSED" if result.refused else "OK"
        print(f"Status  : {status}")
        if result.verdict:
            print(f"Stars   : {result.verdict.star_rating} | Confidence: {result.verdict.confidence.value}")
            print(f"\nEN: {result.verdict.summary_en[:120]}…")
            print(f"AR: {result.verdict.summary_ar[:120]}…")
        else:
            print(f"Refusal : {result.refusal_reason}")
        return result.model_dump()
    except (ValidationError, json.JSONDecodeError) as e:
        print(f"SCHEMA ERROR: {e}")
        print(f"Raw output was:\n{raw[:500]}")
        return {"error": str(e), "raw": raw, "product_id": product_id}


def main():
    parser = argparse.ArgumentParser(description="Run the Moms Verdict pipeline")
    parser.add_argument("--product", help="Product ID to run")
    parser.add_argument("--all", action="store_true", help="Run all products")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model string for OpenRouter")
    args = parser.parse_args()

    data_path = Path("data/reviews.json")
    if not data_path.exists():
        print("Reviews data not found — run: python src/generate_reviews.py")
        sys.exit(1)

    with open(data_path, encoding="utf-8") as f:
        all_data = json.load(f)

    results = {}

    if args.all:
        product_ids = [p["id"] for p in all_data["products"]]
    elif args.product:
        product_ids = [args.product]
    else:
        parser.print_help()
        sys.exit(1)

    for pid in product_ids:
        results[pid] = run_product(pid, all_data, args.model)

    out_path = Path("data/verdicts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Verdicts written to {out_path}")


if __name__ == "__main__":
    main()
