"""Behavior checks for the actual functions defined in the submitted notebooks."""
import ast
import json
import re
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def notebook_function(lab, name, **namespace):
    notebook = json.loads(next((ROOT / f"Lab{lab}").glob("*.ipynb")).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] != "code" or f"def {name}(" not in source:
            continue
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
        if functions:
            scope = {"re": re, "np": np, **namespace}
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(ROOT), "exec"), scope)
            return scope[name]
    raise AssertionError(f"Lab {lab}: required solution function {name} is missing")


class SolutionTests(unittest.TestCase):
    def test_hashtags_count_repetitions_and_unicode(self):
        extract = notebook_function(2, "extract_hashtags")
        self.assertEqual(extract("#Apple #apple #iPhone14 #NLP_2026 #السعودية"),
                         ["apple", "apple", "iphone14", "nlp_2026", "السعودية"])
        self.assertEqual(extract(None), [])
        self.assertEqual(extract("plain text and a#fragment"), [])

    def test_tweet_cleaning_removes_requested_noise(self):
        clean = notebook_function(3, "clean_tweet")
        import emoji
        clean.__globals__["emoji"] = emoji
        self.assertEqual(clean("RT @user: Pakistan is great! https://example.com #news 😀"),
                         "pakistan is great!")
        self.assertEqual(clean(None), "")
        self.assertEqual(clean("partyRT is here"), "partyrt is here")
        self.assertEqual(clean("Pakistan is\ufffd great"), "pakistan is great")

    def test_amazon_preprocessing_preserves_negation(self):
        clean = notebook_function(4, "preprocess_amazon")
        import nltk
        nltk.data.path.insert(0, str(ROOT / ".cache" / "nltk_data"))
        from nltk.stem import WordNetLemmatizer
        clean.__globals__.update(STOP_WORDS={"this", "is", "a", "the", "and"},
                                 lemmatizer=WordNetLemmatizer())
        self.assertEqual(clean("This is NOT bad!!!"), "not bad")
        self.assertEqual(clean(""), "")
        self.assertEqual(clean("Phones and batteries"), "phone battery")

    def test_simpsons_cleaning_matches_instructor_procedure(self):
        clean = notebook_function(5, "clean_text")
        self.assertEqual(clean('Homer (123), “Marge”.'), "homer  marge")

    def test_review_vectors_average_known_words_and_handle_empty_reviews(self):
        vectorize = notebook_function(6, "review_to_vector")
        vectors = {"good": np.ones(200, dtype=np.float32),
                   "great": np.full(200, 3, dtype=np.float32)}
        np.testing.assert_allclose(vectorize(["good", "unknown", "great"], vectors, 200),
                                   np.full(200, 2, dtype=np.float32))
        np.testing.assert_array_equal(vectorize([], vectors, 200), np.zeros(200))
        np.testing.assert_array_equal(vectorize(["unknown"], vectors, 200), np.zeros(200))


if __name__ == "__main__":
    unittest.main()
