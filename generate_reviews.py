"""
Generate synthetic reviews for 3 baby products.

Products:
  - Babybjorn Bouncer Balance Soft  (~80 reviews, mostly positive, EN+AR)
  - Medela Freestyle Flex Pump      (~80 reviews, mixed, EN+AR)
  - Generic Brand Teething Ring     (~15 reviews — triggers LOW confidence path)

Also generates 5 adversarial edge-case batches for evals.

Writes to: data/reviews.json
"""

import json, os, random, sys
from pathlib import Path

# Ensure Unicode logs render on Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Simple deterministic synthetic data (no API call needed for data gen) ──
# This keeps the project self-contained and avoids API costs for data.
# The LLM call happens only in the synthesis step (pipeline.py).

PRODUCTS = [
    {
        "id": "babybjorn-bouncer",
        "name": "Babybjorn Bouncer Balance Soft",
        "category": "bouncers",
        "true_star": 4.3,
    },
    {
        "id": "medela-freestyle",
        "name": "Medela Freestyle Flex Breast Pump",
        "category": "feeding",
        "true_star": 3.8,
    },
    {
        "id": "generic-teething",
        "name": "NoBrand Silicone Teething Ring",
        "category": "teething",
        "true_star": 3.2,
        "review_count": 15,   # LOW confidence trigger
    },
]

# Seed phrases used to build reviews — varied enough, not repetitive
BOUNCER_EN_POS = [
    "Absolute lifesaver! My 3-month-old calmed down instantly.",
    "The gentle bounce motion is perfect — she falls asleep in minutes.",
    "Well worth the price. Solid build quality, easy to fold.",
    "My baby loves it. We've used it every day for 4 months.",
    "Very easy to assemble. The fabric is soft and washable.",
    "Best purchase we made for the nursery. Highly recommend.",
    "Light enough to carry room to room. Really practical.",
    "My son has reflux and this angle is perfect for him.",
    "Looks beautiful and feels premium. Totally justifies the cost.",
    "The bouncer grows with the baby — we'll use it for years.",
]
BOUNCER_EN_NEG = [
    "Expensive for what it is. Wish it had a vibration mode.",
    "My baby outgrew it faster than expected.",
    "The fabric pilled after washing — a bit disappointing.",
    "Folds awkwardly. Hard to store in a small flat.",
    "No newborn insert — had to buy that separately.",
]
BOUNCER_EN_NEU = [
    "Decent bouncer, does what it says. Nothing extraordinary.",
    "Average quality for the price range.",
    "It's fine. Baby uses it occasionally.",
]
BOUNCER_AR_POS = [
    "رائعة جداً! طفلتي تنام فيها بسرعة.",
    "أفضل شراء عملته للطفل. جودة ممتازة.",
    "خفيفة وسهلة الحمل. أنصح بها بشدة.",
    "القماش ناعم جداً ومريح للطفل.",
    "تستحق كل ريال دفعته فيها.",
    "تصميم أنيق ومتين. راضية تماماً.",
    "ابني يهدأ فوراً حين أضعه فيها.",
]
BOUNCER_AR_NEG = [
    "سعرها مرتفع بعض الشيء.",
    "توقعت جودة أعلى لهذا السعر.",
    "القماش يتكرمش بعد الغسيل.",
]

MEDELA_EN_POS = [
    "Portable and powerful. I can pump hands-free at my desk.",
    "The app connectivity is genuinely useful — tracks sessions automatically.",
    "Quiet enough to use at night without waking anyone.",
    "My supply increased within a week of using this consistently.",
    "Comfortable flanges, easy to clean, reliable motor.",
    "Battery life is impressive — lasts a full work day.",
    "Finally a pump that doesn't feel like a chore to set up.",
]
MEDELA_EN_NEG = [
    "The app crashes occasionally on Android. Frustrating.",
    "Suction weaker than my old Spectra. Disappointed.",
    "One of the membranes cracked after 3 weeks.",
    "Instructions are confusing. Took an hour to figure out.",
    "Expensive replacement parts. Adds up quickly.",
    "Louder than advertised. Audible in a quiet room.",
    "Flange sizing limited — had to buy third-party ones.",
]
MEDELA_EN_NEU = [
    "Works as expected. Nothing special but gets the job done.",
    "Similar to other Medela products I've used.",
    "Average suction for the price.",
]
MEDELA_AR_POS = [
    "مريحة جداً واستخدامها بسيط.",
    "الإنتاج تحسّن بعد أسبوع من الاستخدام.",
    "هادئة نسبياً، لا توقظ الطفل.",
    "عملية جداً للأمهات العاملات.",
]
MEDELA_AR_NEG = [
    "التطبيق به أعطال متكررة.",
    "قوة الشفط أقل من المتوقع.",
    "قطع الغيار غالية جداً.",
    "التعليمات غير واضحة.",
]

TEETHING_EN_POS = [
    "Baby loves chewing on it. Seems safe.",
    "Good for the price. Does the job.",
    "Easy to clean in the dishwasher.",
]
TEETHING_EN_NEG = [
    "Broke after two weeks. Cheap plastic.",
    "Worried about the material quality — no safety cert mentioned.",
    "Too small, baby dropped it constantly.",
    "Strong chemical smell when new.",
    "Not BPA-free as claimed? Unclear.",
]
TEETHING_AR_NEG = [
    "الجودة سيئة، انكسرت بسرعة.",
    "رائحة كيميائية غير مريحة.",
]


def make_reviews(product, pos_en, neg_en, neu_en, pos_ar, neg_ar, count=80):
    reviews = []
    stars = product["true_star"]
    total = product.get("review_count", count)

    # Distribution: 50% pos, 25% neg, 25% neu for good products; adjusted per product
    for i in range(total):
        roll = random.random()
        if roll < 0.55:
            text = random.choice(pos_en)
            lang = "en"
            star = random.choice([4, 5])
        elif roll < 0.75:
            text = random.choice(neg_en)
            lang = "en"
            star = random.choice([1, 2, 3])
        elif roll < 0.85:
            text = random.choice(neu_en) if neu_en else random.choice(pos_en)
            lang = "en"
            star = 3
        elif roll < 0.93:
            text = random.choice(pos_ar)
            lang = "ar"
            star = random.choice([4, 5])
        else:
            text = random.choice(neg_ar)
            lang = "ar"
            star = random.choice([1, 2])

        reviews.append({
            "product_id": product["id"],
            "review_id": f"{product['id']}-r{i:03d}",
            "lang": lang,
            "star": star,
            "text": text,
            "verified_purchase": random.random() > 0.15,
        })

    return reviews


def main():
    random.seed(42)
    all_reviews = []

    all_reviews += make_reviews(PRODUCTS[0], BOUNCER_EN_POS, BOUNCER_EN_NEG, BOUNCER_EN_NEU, BOUNCER_AR_POS, BOUNCER_AR_NEG, 80)
    all_reviews += make_reviews(PRODUCTS[1], MEDELA_EN_POS, MEDELA_EN_NEG, MEDELA_EN_NEU, MEDELA_AR_POS, MEDELA_AR_NEG, 80)
    all_reviews += make_reviews(PRODUCTS[2], TEETHING_EN_POS, TEETHING_EN_NEG, [], [], TEETHING_AR_NEG, 15)

    # Edge case: product with 0 reviews (for INSUFFICIENT confidence eval)
    PRODUCTS.append({
        "id": "mystery-product",
        "name": "Unknown Brand Baby Monitor",
        "category": "monitors",
        "true_star": None,
    })

    out = {
        "products": PRODUCTS,
        "reviews": all_reviews,
        "edge_cases": {
            "zero_reviews_product_id": "mystery-product",
            "low_review_product_id": "generic-teething",
            "mixed_signal_product_id": "medela-freestyle",
            "spam_reviews": [
                {"product_id": "babybjorn-bouncer", "review_id": "spam-001", "lang": "en", "star": 5,
                 "text": "BEST PRODUCT EVER BUY NOW!!!! 10/10 AMAZING DEAL", "verified_purchase": False},
                {"product_id": "babybjorn-bouncer", "review_id": "spam-002", "lang": "en", "star": 5,
                 "text": "Perfect perfect perfect perfect perfect perfect", "verified_purchase": False},
            ],
            "contradictory_reviews": [
                {"product_id": "medela-freestyle", "review_id": "contra-001", "lang": "en", "star": 5,
                 "text": "Absolutely silent — you can't hear it at all!", "verified_purchase": True},
                {"product_id": "medela-freestyle", "review_id": "contra-002", "lang": "en", "star": 2,
                 "text": "Way too loud, woke up the whole house every session.", "verified_purchase": True},
            ],
        }
    }

    Path("data").mkdir(exist_ok=True)
    with open("data/reviews.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✓ Generated {len(all_reviews)} reviews across {len(PRODUCTS)} products")
    print(f"  • {PRODUCTS[0]['id']}: {sum(1 for r in all_reviews if r['product_id'] == PRODUCTS[0]['id'])} reviews")
    print(f"  • {PRODUCTS[1]['id']}: {sum(1 for r in all_reviews if r['product_id'] == PRODUCTS[1]['id'])} reviews")
    print(f"  • {PRODUCTS[2]['id']}: {sum(1 for r in all_reviews if r['product_id'] == PRODUCTS[2]['id'])} reviews (LOW confidence)")
    print(f"  • mystery-product: 0 reviews (INSUFFICIENT confidence)")


if __name__ == "__main__":
    main()
