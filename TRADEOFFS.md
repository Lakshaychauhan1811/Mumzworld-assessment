# TRADEOFFS.md — Moms Verdict

## Why this problem

Review synthesis is the highest-leverage AI task I could find for Mumzworld in 5 hours. It maps to a real purchase decision moment — a mother reading 200 reviews before buying a baby bouncer — and it has a clear, provable eval story: ground truth is the reviews themselves. If the verdict says something no review says, it's wrong. That's a testable criterion, not a vibe.

I used generated synthetic data instead of scraping retailer pages, matching the assignment constraint.

I chose it over the other examples in the brief for specific reasons:

- **Gift finder**: Good creative problem but output quality is hard to evaluate objectively. No ground truth.
- **Duplicate product detection**: Strong engineering problem, but less legible as a Loom demo and less customer-facing.
- **Pediatric symptom triage**: High leverage but the safety stakes of a wrong answer are severe. Would require medical validation I can't credibly provide in 5 hours. I rejected this explicitly because I didn't want to build something that *looks* safe but isn't.
- **Customer service email triage**: Too close to a generic classification example. Less opportunity to show multilingual synthesis quality.

---

## Model and architecture choices

### Structured prompt over function calling
I embedded the full JSON schema directly in the system prompt rather than using tool/function calling. Reason: free-tier models on Groq and OpenRouter don't reliably follow tool definitions, but they do follow strict JSON contracts when those contracts are part of the prompt. Tested both approaches; embedded schema won on reliability.

### Pydantic as a hard validation gate
The pipeline does not surface output that fails schema validation — it raises an error. This is deliberate. Silent schema failures (empty strings, null fields that should be populated) are worse than crashes because they look correct. A crash is visible; a silent failure ships to production.

### Temperature 0.3
Review synthesis is a factual task. The model should summarise what reviewers said, not invent new framings. Lower temperature reduces hallucination at the cost of some Arabic copy creativity — acceptable tradeoff for this use case.

### Confidence thresholds: ≥20 / 5–19 / <5 / 0
These thresholds are arbitrary but defensible starting points. In a real deployment, I'd tune them against conversion data: does LOW-confidence verdict suppress purchases more than it should? Are HIGH-confidence verdicts trusted too much for niche products?

### LLM-as-judge for Arabic evaluation
I can't manually verify Gulf Arabic fluency. Rather than ignoring this or claiming the Arabic is good, I added a second LLM call that acts as a judge — scoring fluency, detecting translation-likeness, and checking semantic consistency with the English. This is a known pattern (LLM-as-judge) used in production eval systems. The judge itself can have blind spots in Gulf Arabic idioms, which I noted as a limitation.

### Human-in-the-loop for CS escalation
The review reply agent never auto-sends replies. All tickets land in a queue with `status: pending_human_approval`. This is not a technical limitation — it's an intentional product decision. For baby product safety issues (broken zippers, delivery damage), the cost of a wrong auto-reply is higher than the cost of requiring a human to approve.

### Retry/backoff over flaky free-tier behavior
During full-suite runs, free-tier endpoints can return `429` rate limits. I added retry/backoff and explicit console messaging so the system degrades gracefully and the operational constraint is visible during demos/evals.

---

## What I cut

**Embeddings-based theme clustering**: The brief listed RAG as a desirable AI technique. I scaffolded this in the architecture but fell back to prompt-based theme extraction for the 5-hour scope. Adding sentence-transformers + k-means would improve theme coherence and reduce the chance of the model inventing theme labels. This is the highest-value next step.

**A Gradio or Streamlit UI**: Would make the Loom more visual. Prioritised clean CLI output and correct behaviour over cosmetics.

**Multi-sample confidence calibration**: The model's confidence scores vary between runs. A production system would sample 3–5 times and take a majority vote. Skipped for time — documented as a known limitation.

**Fine-tuning or LoRA**: Rejected explicitly. Llama 3.3 70B already knows Arabic and e-commerce language. Fine-tuning requires (input, ideal output) training pairs I don't have. Adding LoRA without training data would not improve results and would significantly increase complexity without benefit.

**MCP-based parallel processing**: Rejected explicitly. 200 reviews fit comfortably in a single LLM context window. Parallel chunking and distributed orchestration solve problems this dataset doesn't have. Adding MCP here would be overengineering, not engineering.

---

## Uncertainty handling decisions

- **Zero reviews**: Hard refusal. `refused=true`, no verdict generated. The system will not invent a verdict.
- **<5 reviews**: `confidence=low`. Verdict generates but `uncertainty_note` is required by a Pydantic `model_validator` — the model cannot return LOW confidence without explicitly flagging it to the reader.
- **5–19 reviews**: `confidence=medium`. Verdict generates without a required uncertainty note, but the confidence level is visible in the output.
- **≥20 reviews**: `confidence=high`. Full verdict.

The validator enforces this contract — it's not a prompt instruction the model can ignore. If the model returns LOW confidence without an uncertainty note, the Pydantic schema raises a `ValidationError` and the output is rejected.

---

## Known failure modes

1. **Spam reviews (adv-01)**: The model trusts star ratings even for unverified reviews. A batch of all-5★ spam sometimes returns high positive sentiment. Fix: explicitly surface the verified/unverified ratio in the prompt and instruct the model to weight accordingly.

2. **Arabic translation detection**: The LLM judge catches obvious word-for-word translations but may miss subtler cases where sentence structure is English-like but individual words are Arabic. No automated eval is a perfect substitute for a native speaker reviewer.

3. **Short review corpora noise (<10 reviews)**: Theme extraction is unreliable with few data points. The `uncertainty_note` surfaces this, but the themes themselves may not be meaningful. A production system would suppress theme extraction below a minimum count.

4. **Grounding heuristic fragility**: The `check_no_invented_facts` eval uses word-overlap, which is too permissive — a claim passes if it shares any long word with any review. This means it catches obvious hallucinations but not subtle ones. Acknowledged as a limitation; a stronger grounding eval would require reference-based NLI (natural language inference).

---

## What I'd build next (priority order)

1. **Embedding-based theme clustering** — sentence-transformers + k-means for more reliable evidence attribution. Directly fixes the grounding fragility.
2. **Confidence calibration via multi-sample voting** — run synthesis 3 times, aggregate. Reduces per-run variance.
3. **CS dashboard UI** — approve/reject queue for the escalation agent. Makes HITL actually usable by non-technical CS staff.
4. **Webhook integration** — connect escalation queue to Zendesk or Freshdesk. Makes the agent production-deployable.
5. **Verified purchase weighting in prompt** — directly fixes adv-01 spam failure.
