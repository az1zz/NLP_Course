# Verified results

**Student:** Abdulaziz Saqer BinRuqush

**Student ID:** 2230007229

All eight notebooks were executed in fresh Python kernels on CPU. All **129
code cells** completed with **zero execution errors**. Five behavior tests
passed, and `pip check` found no incompatible installed dependencies.
The saved notebook outputs contain no warning streams.

| Lab | Executed code cells | Selected measured result |
| --- | ---: | --- |
| 1 | 10 | Loaded all 50,000 IMDB reviews and inspected the data |
| 2 | 32 | 1,615 hashtag occurrences; 555 distinct case-insensitive hashtags |
| 3 | 16 | 129,058 unique cleaned tweets; P(is \| pakistan) = 0.04310 |
| 4 | 10 | Amazon Naive Bayes test accuracy: 79.56% |
| 5 | 20 | 100-dimensional skip-gram embeddings; 131,839 nonempty Simpsons lines |
| 6 | 14 | TF-IDF network accuracy: 76%; Word2Vec network accuracy: 58.5% |
| 7 | 10 | Ran all three transformer families, sentiment tasks, summarization, and sampling |
| 8 | 17 | Contextual POS tags, comparison table, and saved displaCy diagrams |

## Interpretation

Lab 3 uses the required unsmoothed MLE model. Its held-out perplexity is
infinite because 169,287 of 512,601 test bigrams have zero probability;
observed-event coverage is 66.97%. The notebook explains both the unigram
word-perplexity and the boundary-padded one-word sentence interpretation.

Lab 4's Amazon model has weak neutral-class recall. Its full classification
report and confusion matrix show that limitation alongside overall accuracy.
Lab 6's required Word2Vec network performs worse than its TF-IDF baseline;
its architecture, learning rate, and 15 epochs are preserved. Embedding
standardization is fitted on training reviews only.

Lab 7 records actual model-generated text. FLAN-T5-small's French translation
is imperfect, and its summary is brief and omits the paragraph's distinctions
between model families. These outputs demonstrate the small model's limits.

## Execution evidence

[execution_report.json](execution_report.json) records each notebook's cell
count, execution time, timestamp, and current SHA-256. It also retains the
original execution-file hash: after execution, notebook schema metadata and
cell IDs were normalized to nbformat 4.5. Code, markdown, execution counts,
and outputs were checked to remain unchanged by that normalization.

`scripts/verify_submission.py` validates all eight saved files, hashes,
student identity, execution counts, and required image/SVG outputs.
See the [course README](README.md) for reproduction commands.
