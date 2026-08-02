from src.cleaner import clean_text


def test_clean_basic_lowercase():
    assert clean_text("HELLO World!") == "hello world"


def test_clean_remove_html_and_url():
    s = "<p>Please visit http://example.com now!</p>"
    assert "http" not in clean_text(s)
    assert "<p>" not in clean_text(s)


def test_preserve_negation():
    s = "I am NOT happy and I don't want this"
    out = clean_text(s)
    assert "not" in out or "don't" in out or "dont" in out


def test_empty_and_nonstring():
    assert clean_text(123) == ""
    assert clean_text("") == ""
