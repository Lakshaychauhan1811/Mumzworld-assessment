"""
Review Reply Agent — Human-in-the-Loop CS Escalation

For each review, the agent:
  1. Classifies the intent and severity
  2. Drafts a bilingual customer-facing reply (EN + AR)
  3. For serious issues (quality, delivery damage):
       → Creates an escalation ticket routed to a CS employee
       → Never auto-replies — human must approve first
  4. For positive reviews:
       → Drafts a thank-you reply with a soft upsell nudge
  5. All decisions are logged to escalations/queue.json for human review

This is a deliberate Human-in-the-Loop design:
  - AI classifies and drafts
  - Human approves before anything reaches the customer
  - Escalation tickets include full context so CS can act without re-reading

Usage:
  python src/review_agent.py --product babybjorn-bouncer
  python src/review_agent.py --all
  python src/review_agent.py --review "The zipper broke after 2 days, terrible quality"
"""

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import call_llm


# ── Enums & Schema ────────────────────────────────────────────────────────────

class ReviewCategory(str, Enum):
    QUALITY_ISSUE    = "quality_issue"      # Defective, broke, poor material
    DELIVERY_DAMAGE  = "delivery_damage"    # Arrived damaged, wrong item, lost
    SHIPPING_DELAY   = "shipping_delay"     # Late delivery, not arrived
    POSITIVE         = "positive"           # Happy customer, recommends
    GENERAL_NEGATIVE = "general_negative"   # Disappointed but not actionable
    OTHER            = "other"


class EscalationPriority(str, Enum):
    HIGH   = "high"    # Quality/damage — needs CS action within 24h
    MEDIUM = "medium"  # Shipping delay — needs tracking check
    LOW    = "low"     # General negative — monitor only
    NONE   = "none"    # Positive — no escalation needed


class ReviewClassification(BaseModel):
    category: ReviewCategory
    priority: EscalationPriority
    confidence: float = Field(..., ge=0.0, le=1.0)
    requires_proof: bool = Field(
        ...,
        description="True if CS should ask customer for photo/video proof (quality/damage issues)"
    )
    requires_escalation: bool = Field(
        ...,
        description="True if a human CS employee must review before any action"
    )
    issue_summary: str = Field(
        ...,
        description="One-sentence summary of the customer's issue for the CS ticket"
    )


class DraftReply(BaseModel):
    reply_en: str = Field(..., description="Customer-facing reply in English")
    reply_ar: str = Field(..., description="Customer-facing reply in Arabic (native, not translated)")
    internal_note: str = Field(
        ...,
        description="Internal note for CS employee — what action is needed, what to check"
    )


class EscalationTicket(BaseModel):
    ticket_id: str
    created_at: str
    product_id: str
    product_name: str
    review_id: str
    review_text: str
    review_star: int
    review_lang: str
    classification: ReviewClassification
    draft_reply: DraftReply
    status: str = "pending_human_approval"
    assigned_to: str = "cs-team@mumzworld.com"
    action_required: str  # Clear instruction for the CS employee


# ── Prompts ───────────────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM = """You are a customer service AI for Mumzworld, the largest baby e-commerce platform in the Middle East.

Classify a product review and decide how to handle it.

Categories:
- quality_issue: Product is defective, broke, poor materials, safety concern
- delivery_damage: Item arrived damaged, wrong item sent, package lost
- shipping_delay: Late delivery, still waiting, not arrived on time
- positive: Customer is happy, recommending, leaving praise
- general_negative: Disappointed but no specific actionable issue
- other: Doesn't fit above

Priority:
- high: quality_issue or delivery_damage — CS must act within 24h
- medium: shipping_delay — needs tracking check
- low: general_negative — monitor only
- none: positive or other

Escalation rules:
- requires_proof: true only for quality_issue and delivery_damage (ask for photo/video)
- requires_escalation: true for high and medium priority

Respond ONLY with valid JSON:
{
  "category": "string",
  "priority": "string",
  "confidence": float 0-1,
  "requires_proof": boolean,
  "requires_escalation": boolean,
  "issue_summary": "one sentence"
}"""

REPLY_SYSTEM = """You are a warm, professional customer service agent for Mumzworld, a baby e-commerce platform serving mothers across the Middle East.

Write a bilingual reply (English + Arabic) to a customer review.

Tone rules:
- Always empathetic and respectful — these are mothers dealing with baby products
- For issues: acknowledge the problem, explain the next step clearly, never make promises you can't keep
- For quality/damage issues: ask for photo or video proof politely, explain it helps resolve faster
- For positive reviews: thank genuinely (not generic), add a soft relevant product nudge
- Arabic must be native Gulf Arabic copy — NOT a translation of the English
- Keep replies concise: 3–4 sentences max per language

Also write an internal_note for the CS employee — what they need to do, what to check in the system.

Respond ONLY with valid JSON:
{
  "reply_en": "string",
  "reply_ar": "string",
  "internal_note": "string"
}"""


# ── Core functions ────────────────────────────────────────────────────────────

def classify_review(review_text: str, star: int, lang: str) -> ReviewClassification:
    user_prompt = f"""Review ({lang.upper()}, {star}★):
{review_text}

Classify this review."""

    raw = call_llm(CLASSIFIER_SYSTEM, user_prompt)
    cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return ReviewClassification.model_validate(json.loads(cleaned))


def draft_reply(
    review_text: str,
    star: int,
    classification: ReviewClassification,
    product_name: str,
) -> DraftReply:
    action_context = ""
    if classification.requires_proof:
        action_context = "This is a quality/damage issue. Ask for photo or video proof."
    elif classification.category == ReviewCategory.POSITIVE:
        action_context = "This is a positive review. Thank the customer warmly and suggest a complementary product."
    elif classification.category == ReviewCategory.SHIPPING_DELAY:
        action_context = "This is a shipping delay. Apologize and assure them CS will check the order status."
    else:
        action_context = "General negative review. Acknowledge and invite them to contact CS."

    user_prompt = f"""Product: {product_name}
Review ({star}★): {review_text}
Classification: {classification.category.value} | Priority: {classification.priority.value}
Issue summary: {classification.issue_summary}
Context: {action_context}

Draft a bilingual customer reply and an internal CS note."""

    raw = call_llm(REPLY_SYSTEM, user_prompt)
    cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return DraftReply.model_validate(json.loads(cleaned))


def determine_action_required(classification: ReviewClassification) -> str:
    if classification.category == ReviewCategory.QUALITY_ISSUE:
        return "REQUEST PROOF: Contact customer, ask for photos/video of defect. If confirmed, initiate replacement or refund per policy."
    elif classification.category == ReviewCategory.DELIVERY_DAMAGE:
        return "REQUEST PROOF: Ask customer for photos of damaged packaging/item. Check order in logistics system. Initiate claim with courier."
    elif classification.category == ReviewCategory.SHIPPING_DELAY:
        return "CHECK TRACKING: Look up order in system. If delayed beyond SLA, proactively offer compensation. Update customer."
    elif classification.category == ReviewCategory.POSITIVE:
        return "NO ACTION REQUIRED: Positive review. Draft reply is ready to send — approve if tone is appropriate."
    else:
        return "MONITOR: Log issue. Reply when ready. No urgent action required."


def process_review(
    review: dict,
    product_name: str,
    product_id: str,
) -> EscalationTicket:
    """Full pipeline: classify → draft reply → create ticket."""

    classification = classify_review(
        review_text=review["text"],
        star=review["star"],
        lang=review.get("lang", "en"),
    )

    reply = draft_reply(
        review_text=review["text"],
        star=review["star"],
        classification=classification,
        product_name=product_name,
    )

    ticket = EscalationTicket(
        ticket_id=f"MWZ-{uuid.uuid4().hex[:8].upper()}",
        created_at=datetime.utcnow().isoformat() + "Z",
        product_id=product_id,
        product_name=product_name,
        review_id=review.get("review_id", "unknown"),
        review_text=review["text"],
        review_star=review["star"],
        review_lang=review.get("lang", "en"),
        classification=classification,
        draft_reply=reply,
        action_required=determine_action_required(classification),
    )

    return ticket


# ── Queue manager ─────────────────────────────────────────────────────────────

def save_to_queue(tickets: list[EscalationTicket], queue_path: Path):
    """Append tickets to the human review queue."""
    existing = []
    if queue_path.exists():
        with open(queue_path, encoding="utf-8") as f:
            existing = json.load(f)

    new_entries = [t.model_dump() for t in tickets]
    all_entries = existing + new_entries

    # Sort: high priority first
    priority_order = {"high": 0, "medium": 1, "low": 2, "none": 3}
    all_entries.sort(key=lambda x: priority_order.get(
        x["classification"]["priority"], 99
    ))

    with open(queue_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    return len(new_entries)


def print_ticket(ticket: EscalationTicket):
    cat = ticket.classification.category.value
    pri = ticket.classification.priority.value
    icon = {"high": "🔴", "medium": "🟡", "low": "🔵", "none": "🟢"}.get(pri, "⚪")
    print(f"\n  {icon} [{ticket.ticket_id}] {cat.upper()} | {pri} priority")
    print(f"     Review  : {ticket.review_star}★ — {ticket.review_text[:70]}…")
    print(f"     Issue   : {ticket.classification.issue_summary}")
    print(f"     Action  : {ticket.action_required}")
    print(f"     Reply EN: {ticket.draft_reply.reply_en[:80]}…")
    print(f"     Reply AR: {ticket.draft_reply.reply_ar[:60]}…")
    if ticket.classification.requires_proof:
        print(f"     ⚠️  Proof required from customer")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Review Reply Agent")
    parser.add_argument("--product", help="Process reviews for a specific product ID")
    parser.add_argument("--all", action="store_true", help="Process all products")
    parser.add_argument("--review", help="Process a single review text directly")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max reviews to process per product (default: 10)")
    parser.add_argument("--queue", default="escalations/queue.json",
                        help="Path to the human review queue file")
    args = parser.parse_args()

    queue_path = Path(args.queue)
    queue_path.parent.mkdir(exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  Mumzworld Review Reply Agent")
    print(f"  Human-in-the-Loop CS Escalation")
    print(f"{'═'*60}")

    tickets = []

    # Single review mode
    if args.review:
        review = {
            "text": args.review,
            "star": 1,
            "lang": "en",
            "review_id": "cli-input",
        }
        print(f"\n  Processing: \"{args.review[:60]}…\"")
        ticket = process_review(review, "Unknown Product", "cli-input")
        print_ticket(ticket)
        tickets.append(ticket)

    else:
        data_path = Path("data/reviews.json")
        if not data_path.exists():
            print("Run: python src/generate_reviews.py first")
            sys.exit(1)

        with open(data_path, encoding="utf-8") as f:
            all_data = json.load(f)

        product_ids = (
            [args.product] if args.product
            else [p["id"] for p in all_data["products"] if p["id"] != "mystery-product"]
        )

        for pid in product_ids:
            product = next((p for p in all_data["products"] if p["id"] == pid), None)
            if not product:
                continue

            reviews = [r for r in all_data["reviews"] if r["product_id"] == pid]

            # Prioritise negative reviews for processing (most actionable)
            negative = [r for r in reviews if r["star"] <= 2]
            positive = [r for r in reviews if r["star"] >= 4]
            sample   = (negative + positive)[:args.limit]

            print(f"\n  Product: {product['name']}")
            print(f"  Processing {len(sample)} reviews ({len(negative)} negative, {len(positive)} positive, capped at {args.limit})…")

            for review in sample:
                try:
                    ticket = process_review(review, product["name"], pid)
                    print_ticket(ticket)
                    tickets.append(ticket)
                except Exception as e:
                    print(f"\n  ERROR on review {review.get('review_id')}: {e}")

    # Save to queue
    if tickets:
        saved = save_to_queue(tickets, queue_path)
        high   = sum(1 for t in tickets if t.classification.priority == EscalationPriority.HIGH)
        medium = sum(1 for t in tickets if t.classification.priority == EscalationPriority.MEDIUM)
        pos    = sum(1 for t in tickets if t.classification.category == ReviewCategory.POSITIVE)

        print(f"\n{'─'*60}")
        print(f"  ✓ {saved} tickets saved to {queue_path}")
        print(f"  🔴 High priority (quality/damage) : {high}")
        print(f"  🟡 Medium priority (shipping)     : {medium}")
        print(f"  🟢 Positive (thank + upsell)      : {pos}")
        print(f"  ⚠️  ALL tickets pending human approval — nothing auto-sent")
        print(f"{'═'*60}\n")
