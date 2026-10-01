"""Export the app's vocabulary as a CSV study sheet without changing app files."""

import argparse
from pathlib import Path
import re

import json5
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = {"fr": "French", "es": "Spanish", "de": "German", "ja": "Japanese", "it": "Italian"}


def load_vocabulary(source):
    text = source.read_text(encoding="utf-8")
    match = re.search(r"^const DICT\s*=\s*(\{.*?^\});", text, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError("Could not find the DICT object in dictionary.js")
    # Parse the object as data; do not execute the JavaScript file.
    vocabulary = json5.loads(match.group(1), allow_duplicate_keys=False)
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("The vocabulary must be a non-empty object")
    for word, entry in vocabulary.items():
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(lang), str) or not entry[lang].strip()
            for lang in LANGUAGES
        ):
            raise ValueError(f"Missing or invalid translation for {word!r}")
    return vocabulary


def export_vocabulary(output, language=None):
    vocabulary = load_vocabulary(ROOT / "dictionary.js")
    selected = {language: LANGUAGES[language]} if language else LANGUAGES
    rows = [
        {"English": word, **{label: entry[code] for code, label in selected.items()}}
        for word, entry in vocabulary.items()
    ]
    sheet = pd.DataFrame(rows).sort_values("English").reset_index(drop=True)
    if output.suffix.lower() != ".csv":
        raise ValueError("The output filename must end in .csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacing an existing study sheet.
    # A UTF-8 BOM helps spreadsheet apps display Japanese and accents correctly.
    with output.open("x", encoding="utf-8-sig", newline="") as handle:
        sheet.to_csv(handle, index=False)
    return len(sheet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=LANGUAGES, help="Export one language; default: all five")
    parser.add_argument("--output", type=Path, default=ROOT / "exports" / "vocabulary.csv")
    args = parser.parse_args()
    try:
        count = export_vocabulary(args.output, args.language)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(f"Exported {count} words to {args.output.resolve()}")


if __name__ == "__main__":
    main()
