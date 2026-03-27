from pathlib import Path


FORBIDDEN = [
    " ".join(["Forward", "Analytics"]),
    "".join(["forward", "analytics"]),
    "".join(["F", "A"]) + " " + " ".join(["Data", "Enrichment"]),
    "".join(["One", "Drive"]),
]


def test_forbidden_strings_absent_from_public_repo() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", ".pytest_cache", "__pycache__"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in FORBIDDEN:
            assert token not in text, f"Found forbidden token {token!r} in {path}"
