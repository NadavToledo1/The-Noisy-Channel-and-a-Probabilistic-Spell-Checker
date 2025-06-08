# Noisy Channel and a Probabilistic Spell Checker

## 🔍 Overview

This project implements a **context-sensitive spell checker** using the **noisy channel model**. It combines a language model with confusion matrices to suggest the most likely correction for a given misspelled word based on context.

- Language Model (LM): Evaluates the fluency of sentences using n-gram statistics.

- Error Model: Models typical spelling errors (insertion, deletion, substitution, transposition) using confusion matrices.

The combined system chooses the sentence with the highest posterior probability given the observed (possibly erroneous) sentence.

---
## 📁 Repository Structure

- `big.txt` - Corpus used to train the language model (Gutenberg corpus)
- `code.py` - Core implementation: language model, spell checker, and utilities
- `Evaluation.py` - Evaluation script to test model performance on benchmark datasets
---
## 🎨 Features

- Detects and corrects non-word errors (invalid words)

- Handles real-word errors using contextual sentence analysis

- Combines error and language models via a probabilistic framework

- Supports word-level and character-level language models

- Edit-distance based candidate generation (up to 2 edits)

- Tunable parameter α to balance confidence between original and corrected words
---
## 🔧 Core Components

### code.py

Contains the full implementation of:

SpellChecker: Corrects sentences by evaluating candidate corrections

LanguageModel: Trained on big.txt, computes n-gram probabilities

ErrorModel: Uses error tables to compute likelihood of typos

### spelling_confusion_metrics.py
Includes precomputed confusion matrices for:

- Insertions

- Deletions

- Substitutions

- Transpositions
---
## 📊 Scoring Formula

Each corrected candidate is scored using:
- Score = P(observed | candidate) × P(candidate | context)

P(candidate | context) is computed by the n-gram language model

P(observed | candidate) is given by the error model via confusion matrices

**Words are only replaced when a better-scoring candidate exists.**
---
## 🧰 Error Handling
| Error Type     | Correction Mechanism                                 |
|---------------|---------------------------------------------|
| `Non-word Error`           | Edit-distance + known word filtering            |
| `Real-word Error`     | Sentence scoring with context                  |

If a misspelled or misused word is detected, the system considers alternatives and evaluates which one leads to the most fluent and likely sentence.
---
## 🛠️ Requirements

- Python 3.6+
- nltk
- collections, math, random, re
---
## 🧪 Applications
- Grammar and spell checking in writing tools
- Preprocessing text for NLP tasks
- Educational software for language learning
---
## ✅ Future Improvements
Beam search for candidate pruning

Better real-word error detection

Support for contextual embeddings (e.g., BERT)
---
## ✏️ Author
Nadav Toledo
