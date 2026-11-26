import pytest
from mrai.leaks import find_secret


@pytest.mark.parametrize(
    "text,kind",
    [
        ("-----BEGIN OPENSSH PRIVATE KEY-----\nabc", "private key"),
        ("id=AKIAABCDEFGHIJKLMNOP", "AWS access key"),
        ("t=ghp_" + "a" * 36, "GitHub token"),
        ("k=sk-or-v1-" + "b" * 40, "API key (sk-…)"),
        ("xoxb-1234567890-abc", "Slack token"),
        ("AIza" + "c" * 35, "Google API key"),
        ("eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4", "JWT"),
        (
            "q=%2D%2D%2D%2D%2DBEGIN%20RSA%20PRIVATE%20KEY%2D%2D%2D%2D%2D",
            "private key",
        ),
    ],
)
def test_finds_well_known_credentials(text, kind):
    assert find_secret(text) == kind


def test_finds_the_configured_key_even_url_encoded():
    key = "my-gateway-key/with+chars"
    assert find_secret("https://x.example/?k=my-gateway-key%2Fwith%2Bchars", [key])


@pytest.mark.parametrize(
    "text", [None, "", "https://docs.example/library/re.html?highlight=sk", "skeleton-key"]
)
def test_ordinary_text_passes(text):
    assert find_secret(text, ["short", None]) is None
