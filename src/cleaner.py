import re
import html


NEGATIONS = {"not", "no", "never", "cannot", "can't", "dont", "don't", "n't"}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Unescape HTML entities
    text = html.unescape(text)
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Keep letters, apostrophes (for n't), and whitespace. Replace other symbols with space.
    text = re.sub(r"[^a-z\s']", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    examples = [
        "I am NOT happy. <b>Refund</b> please! Visit http://example.com",
        "Can't login since update #123",
    ]
    for e in examples:
        print(e, "->", clean_text(e))
