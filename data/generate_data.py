import csv
import random
from pathlib import Path


RANDOM_SEED = 42
NUM_SAMPLES = 1000
INTENTS = ["Billing", "Technical Support", "Account Cancellation", "General Inquiry"]


TEMPLATES = {
    "Billing": [
        "I was double charged on my last invoice.",
        "My payment failed but money was taken.",
        "I need a receipt for my recent purchase.",
        "Why was I billed twice this month?",
        "Can I get a refund for the extra charge?",
    ],
    "Technical Support": [
        "My application keeps crashing on startup.",
        "I can't log in even with the right password.",
        "The app shows error code 500 when I save.",
        "Password reset link is not working for me.",
        "I get a blank screen after the update.",
    ],
    "Account Cancellation": [
        "I want to cancel my subscription immediately.",
        "Please close my account and refund the remaining period.",
        "How do I delete my account?",
        "I no longer need this service, cancel my account.",
        "I want to stop my subscription and be refunded.",
    ],
    "General Inquiry": [
        "What are your support hours?",
        "Where can I find the documentation for this feature?",
        "Do you offer discounts for students?",
        "How do I contact sales for enterprise plans?",
        "Is there a roadmap for upcoming features?",
    ],
}


def introduce_edge_cases(text: str, rng: random.Random) -> str:
    # occasionally add negations, HTML, URLs, numbers, or special characters
    if rng.random() < 0.15:
        text = f"I am NOT happy. {text}"
    if rng.random() < 0.1:
        text = text + " Visit http://example.com for details."
    if rng.random() < 0.08:
        text = "<p>" + text + "</p>"
    if rng.random() < 0.05:
        text = text + " Order #12345"
    if rng.random() < 0.04:
        text = text + " !!!"
    return text


def generate_csv(target_path: Path):
    rng = random.Random(RANDOM_SEED)
    rows = []
    per_intent = NUM_SAMPLES // len(INTENTS)
    for intent in INTENTS:
        templates = TEMPLATES[intent]
        for i in range(per_intent):
            base = rng.choice(templates)
            text = introduce_edge_cases(base, rng)
            rows.append({"text": text, "intent": intent})

    # if NUM_SAMPLES not evenly divisible, add remainder to General Inquiry
    while len(rows) < NUM_SAMPLES:
        rows.append({"text": rng.choice(TEMPLATES["General Inquiry"]), "intent": "General Inquiry"})

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "intent"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "customer_intents.csv"
    print(f"Generating synthetic dataset to {out}")
    generate_csv(out)
    print("Done.")
