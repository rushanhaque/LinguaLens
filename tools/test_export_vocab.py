"""Run with: python -m unittest discover -s tools -p 'test_*.py'"""

import csv
from pathlib import Path
import tempfile
import unittest

from export_vocab import export_vocabulary, load_vocabulary


class ExportTests(unittest.TestCase):
    def test_all_languages_and_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "all.csv"
            count = export_vocabulary(output)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, ["English", "French", "Spanish", "German", "Japanese", "Italian"])
            self.assertEqual(count, 80)
            self.assertEqual(len(rows), count)
            self.assertEqual([row["English"] for row in rows], sorted(row["English"] for row in rows))
            by_word = {row["English"]: row for row in rows}
            self.assertEqual(by_word["cat"]["Japanese"], "ねこ (neko)")
            self.assertEqual(by_word["bicycle"]["French"], "vélo")
            self.assertEqual(by_word["fire hydrant"]["French"], "bouche d'incendie")

    def test_single_language_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "ja.csv"
            export_vocabulary(output, "ja")
            original = output.read_bytes()
            with output.open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(csv.DictReader(handle).fieldnames, ["English", "Japanese"])
            with self.assertRaises(FileExistsError):
                export_vocabulary(output, "fr")
            self.assertEqual(output.read_bytes(), original)

    def test_rejects_incomplete_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dictionary.js"
            source.write_text('const DICT = {\ncat: {fr: "chat"}\n};', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "translation"):
                load_vocabulary(source)

    def test_rejects_non_csv_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sheet.xlsx"
            with self.assertRaisesRegex(ValueError, "csv"):
                export_vocabulary(output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
