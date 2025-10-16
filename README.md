# Toxic Comment Classifier (NLTK + scikit-learn)

An end-to-end NLP pipeline to classify short comments as **Toxic (1)** or **Non‑toxic (0)**.

## Tech Stack
- Python
- NLTK (tokenization, stopword removal, lemmatization with fallback to stemming if offline)
- scikit‑learn (TF‑IDF features, Logistic Regression classifier)
- pandas (data handling)

## Dataset
The script uses the 20 provided samples (10 toxic, 10 non‑toxic) as the training corpus.  
Labels: `0 = Non‑toxic`, `1 = Toxic`

## Preprocessing
- Lowercasing
- Punctuation stripping
- Tokenization (whitespace)
- Stopword removal (bundled minimal English list to avoid external downloads)
- Lemmatization via NLTK WordNet if available, else fallback to Porter stemming
- Edge cases:
  - Very short/empty inputs get a placeholder token so the vectorizer never receives an empty feature set.
  - **Non‑English** text: a lightweight heuristic prints a warning that predictions may be unreliable.

## Features
- TF‑IDF with unigrams + bigrams

## Model
- Logistic Regression (`liblinear`, `max_iter=1000`)
- Train/test split (70/30, stratified)
- Metrics: Accuracy, Precision, Recall, F1 + classification report

## How to Run
```bash
python toxic_classifier.py --demo
python toxic_classifier.py --comment "This sucks badly."
python toxic_classifier.py --comment "Great job on the project!"
```

## Sample Output
See `sample_output.txt` for a captured run including evaluation and demo predictions.
