import unicodedata

from src.preprocessing import preprocess, script_group


def test_emoji_kept_and_case_preserved():
    assert preprocess("Climate is REAL 🌍🔥") == "Climate is REAL 🌍🔥"


def test_handles_urls_hashtags_noise():
    t = "RT @greenpeace: Act now https://t.co/abc www.x.org #ClimateAction #SemST"
    assert preprocess(t) == "@USER: Act now URL URL ClimateAction"


def test_email_is_not_a_handle():
    assert preprocess("mail me at a@b.com") == "mail me at a@b.com"


def test_whitespace_collapsed():
    assert preprocess("  a \t b\n\nc  ") == "a b c"


def test_empty_and_missing():
    assert preprocess(None) == ""
    assert preprocess(float("nan")) == ""
    assert preprocess("   #SemST  ") == ""


def test_hindi_bengali_nfc():
    decomposed = unicodedata.normalize("NFD", "क़ानून জলবায়ু")
    out = preprocess(decomposed)
    assert out == unicodedata.normalize("NFC", "क़ानून জলবায়ু")
    assert preprocess("ग्लोबल वार्मिंग एक धोखा है।") == "ग्लोबल वार्मिंग एक धोखा है।"


def test_idempotent():
    t = "RT @x: #Global warming 😡 http://a.b"
    assert preprocess(preprocess(t)) == preprocess(t)


def test_script_group():
    assert script_group("global warming is real") == "latin"
    assert script_group("ग्लोबल वार्मिंग") == "devanagari"
    assert script_group("জলবায়ু পরিবর্তন") == "bengali"
    assert script_group("global warming एक धोखा") == "devanagari"
    assert script_group("") == "latin"
