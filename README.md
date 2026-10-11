# NLP Course — Labs 1–8

**Student:** Abdulaziz Saqer BinRuqush

**Student ID:** 2230007229

Each lab folder contains the instructor's notebook with completed exercises,
written explanations, and saved outputs from a fresh-kernel execution.
Instructor repository URLs and downloaded commit hashes are recorded in
[sources.json](sources.json).

See [verified results and model limitations](RESULTS.md) for measured outcomes
and the final validation record.

| Lab | Topic |
| --- | --- |
| [Lab 1](Lab1/) | Introduction and IMDB CSV exploration |
| [Lab 2](Lab2/) | Preprocessing, hashtag counts, regular expressions, and tokenization |
| [Lab 3](Lab3/) | Pakistan tweets and bigram MLE language modeling |
| [Lab 4](Lab4/) | Disneyland SVM example and Amazon phone-review Naive Bayes |
| [Lab 5](Lab5/) | TF-IDF, cosine similarity, and skip-gram Word2Vec |
| [Lab 6](Lab6/) | Yelp sentiment classification with PyTorch and Word2Vec |
| [Lab 7](Lab7/) | Transformer model families, inference, and reflection |
| [Lab 8](Lab8/) | POS tagging and dependency visualization |

## Reproduce on Windows

The solutions use Python 3.9 and CPU execution. From the repository root in
PowerShell, create an environment and install the pinned packages:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts\prepare_resources.py
.\.venv\Scripts\python.exe scripts\course_data.py
```

Register the environment if you want to use a notebook editor:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name nlp-course --display-name "Python (NLP Course)"
```

Open each notebook with that kernel, starting Jupyter from the repository root
or its lab folder. You can also execute all notebooks without opening an editor:

```powershell
.\.venv\Scripts\python.exe scripts\run_labs.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\verify_submission.py
```

To execute selected labs, pass their numbers, for example:

```powershell
.\.venv\Scripts\python.exe scripts\run_labs.py 6 7 8
```

Every lab starts a new kernel. The runner stops on an error, writes the partial
notebook to ignored local scratch, and only replaces the submitted notebook
after the complete execution succeeds. [execution_report.json](execution_report.json)
records execution time, code-cell count, and the SHA-256 of each verified notebook.
The report also documents the final notebook schema normalization, which leaves
the executed code and outputs unchanged.
The complete tested Windows environment is pinned in `requirements-lock.txt`;
`requirements.txt` lists the direct dependencies. Transformer checkpoint commits
are pinned in [model_revisions.json](model_revisions.json).

## Data and models

The Yelp and Simpsons data supplied in the instructor repositories are included
in their corresponding lab folders. Other data are downloaded from the exact
instructor-linked Kaggle datasets: IMDB, Apple tweets, Disneyland, Amazon phones,
and fake/real news use version 1; Pakistan tweets use version 2.

External datasets, NLTK resources, and Hugging Face model weights are cached in
the ignored `.cache` folder. They are acquired automatically as needed, and are
not redistributed in this repository. Lab 7 downloads DistilGPT-2,
`google/flan-t5-small`, and
`distilbert-base-uncased-finetuned-sst-2-english` on its first run.
Use an internet connection for the initial downloads. Complete-dataset training
and sequential CPU model loading can take some time.

## Evaluation choices

Train/test splits use seed 42, and learned text representations are fitted on
training data only. The full datasets are processed without sampling. Lab 3
removes duplicate cleaned tweets before splitting; Lab 4 also removes duplicate
cleaned Amazon reviews and reviews with conflicting labels to avoid leakage.
Counts of excluded rows are displayed.

The Pakistan CSV contains some malformed UTF-8 bytes. Lab 3 preserves valid
Unicode, counts affected tweets, and removes decoding replacement characters
during cleaning. The source CSV is unchanged.

Lab 3 keeps the instructor-required unsmoothed MLE model. Infinite held-out
perplexity is expected when it encounters a zero-probability event; the notebook
reports unseen-event counts and coverage and explains the result. Lab 7 retains
the instructor's original task numbering, which skips Task 4.
