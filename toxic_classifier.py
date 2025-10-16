"""
toxic_classifier.py
===================
End-to-end toxic vs non-toxic classifier using:
- NLTK for preprocessing (lowercasing, tokenization, stopword removal, lemmatization with safe fallback)
- scikit-learn for TF-IDF features (word + char) and Logistic Regression
- pandas for simple dataset handling

Usage:

  pip install -U nltk scikit-learn pandas numpy

  python toxic_classifier.py --demo
  python toxic_classifier.py --comment "This sucks badly."

"""
import argparse
import re
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

# ---------- NLTK (preprocessing) ----------
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer

# ---------- scikit-learn ----------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)


# 1) Dataset (as provided)


NON_TOXIC = [
    "Great job on the project!", "Thanks for the help.", "This is informative.",
    "I appreciate your input.", "Well done everyone.", "Interesting perspective.",
    "Helpful advice here.", "Positive feedback.", "Good discussion.", "Keep it up!"
]

TOXIC = [
    "This is garbage.", "You're an idiot.", "Shut up already.", "Hate this nonsense.",
    "Total waste of time.", "Annoying and stupid.", "Go away loser.", "Useless crap.",
    "Pathetic attempt.", "Disgusting behavior."
]

LABELS = {0: "Non-toxic", 1: "Toxic"}


# 2) Preprocessing helpers (NLTK) with safe fallbacks

# — Stopwords: auto (NLTK) + fallback minimal list (offline-safe) —

"""When I try to use nltk.data.find("corpora/stopwords"), the model isn't loading; I need to use a minimal set of stopwords instead"""

def _load_stopwords():
    minimal = set( """a an and are as at be by for from has have if in into is it its of on or that the this to was were will 
        with you your i me we our ours they them their theirs he she him her his hers it's who whom what which"""
        .split()
    )
    try:
        from nltk.corpus import stopwords
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            try:
                nltk.download("stopwords", quiet=True)
            except Exception:
                pass
        return set(stopwords.words("english"))
    except Exception:
        return minimal

STOPWORDS = _load_stopwords()

# Strip punctuation but keep apostrophes (for contractions)

_PUNCT_RE = re.compile(r"[^\w\s']+", flags=re.UNICODE)

def _setup_lemmatizer():
    """Try WordNet lemmatizer; if unavailable/offline, fall back to Porter stemmer."""
    try:
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            try:
                nltk.download("wordnet", quiet=True)
            except Exception:
                pass
        lem = WordNetLemmatizer()
        _ = lem.lemmatize("tests")
        return ("lemmatize", lem)
    except Exception:
        return ("stem", PorterStemmer())

_LEMMA_MODE, _LEM_TOOL = _setup_lemmatizer()

def _probably_english(text: str) -> bool:
    """Light heuristic to flag likely non-English input for user awareness."""
    if not text or not text.strip():
        return True
    letters = re.findall(r"[A-Za-z]", text)
    ratio = len(letters) / max(1, len(text))
    has_vowel = re.search(r"[aeiouAEIOU]", text) is not None
    return ratio >= 0.6 and has_vowel

def normalize_tokens(text: str) -> List[str]:
    """Tokenize, lowercase, strip punctuation, remove stopwords, lemmatize (or stem)."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    tokens = text.split()

    cleaned = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    if _LEMMA_MODE == "lemmatize":
        processed = []
        for t in cleaned:
            # noun -> verb pass gives a reasonable simple lemmatization
            t1 = _LEM_TOOL.lemmatize(t, pos="n")
            t2 = _LEM_TOOL.lemmatize(t1, pos="v")
            processed.append(t2)
    else:  # stem fallback
        processed = [_LEM_TOOL.stem(t) for t in cleaned]

    # Edge case: if empty after cleaning, keep at least one token
    if not processed:
        processed = tokens[:1] or ["<empty>"]

    return processed

def warn_if_non_english(text: str) -> None:
    if not _probably_english(text):
        warnings.warn("Input may not be English; prediction could be unreliable.", RuntimeWarning)


# 3) Optional tiny lexicon feature (helps on tiny datasets)

TOXIC_LEXICON = {
    "suck", "sucks", "idiot", "stupid", "crap", "loser",
    "garbage", "disgusting", "pathetic", "hate", "shut", "shut up",
}

def _lexicon_counts(texts):
    def count_one(t: str):
        tl = t.lower()
        return sum(1 for w in TOXIC_LEXICON if w in tl)
    arr = np.array([[count_one(t)] for t in texts])
    return arr

LEXICON_FEAT = Pipeline([
    ("lex", FunctionTransformer(_lexicon_counts, validate=False))
])


# 4) Features & model — Word n-grams + Char n-grams (robust to OOV)


def build_pipeline() -> Pipeline:
    # Word-level TF-IDF with our custom tokenizer (unigrams + bigrams)
    word_vec = TfidfVectorizer(
        tokenizer=normalize_tokens,
        preprocessor=None,
        token_pattern=None,     # required when passing a custom tokenizer
        ngram_range=(1, 2)
    )
    # Character-level TF-IDF (inside word boundaries) to generalize to unseen words
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),     # tri- to 5-grams
        min_df=1
    )
    feats = FeatureUnion([
        ("word", word_vec),
        ("char", char_vec),
        ("lex", LEXICON_FEAT),  # tiny lexicon boost
    ])
    clf = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced"  # helpful for tiny data
    )
    return Pipeline([("feats", feats), ("clf", clf)])


# 5) Training, evaluation, prediction


def prepare_dataframe() -> pd.DataFrame:
    texts = NON_TOXIC + TOXIC
    labels = [0] * len(NON_TOXIC) + [1] * len(TOXIC)
    return pd.DataFrame({"text": texts, "label": labels})

def train_and_evaluate(random_state: int = 42):
    """Train the model, print 5-fold CV F1 for stability + hold-out metrics."""
    df = prepare_dataframe()

    # Cross-validation F1 for stability on tiny data
    cv_model = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = cross_val_score(cv_model, df["text"].values, df["label"].values,
                                cv=cv, scoring="f1")
    print(f"5-fold F1 (mean±std): {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

    # Hold-out split to match the assignment
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].values, df["label"].values,
        test_size=0.3, stratify=df["label"].values, random_state=random_state
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

    print("\n=== Hold-out Evaluation (70/30) ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("\nDetailed report:")
    print(classification_report(y_test, y_pred, target_names=[LABELS[0], LABELS[1]]))

    # Also show what the model predicted on the test set (with probabilities)
    results = pd.DataFrame({
        "text": X_test,
        "true": y_test,
        "pred": y_pred,
        "p_non_toxic": np.round(y_proba[:, 0], 3),
        "p_toxic": np.round(y_proba[:, 1], 3),
    })
    print("Test-set predictions:\n", results.to_string(index=False))
    return model

def predict_comment(model: Pipeline, comment: str) -> Tuple[int, List[float]]:
    warn_if_non_english(comment)
    proba = model.predict_proba([comment])[0]  # [non-toxic, toxic]
    pred = int(np.argmax(proba))
    return pred, proba.tolist()


# 6) CLI

def main():
    parser = argparse.ArgumentParser(
        description="Toxic vs Non-toxic comment classifier (NLTK + TF-IDF + Logistic Regression)"
    )
    parser.add_argument("--comment", type=str, default=None, help="Comment text to classify.")
    parser.add_argument("--demo", action="store_true", help="Run CV + hold-out evaluation and demo predictions.")
    args = parser.parse_args()

    model = train_and_evaluate()

    if args.comment is not None:
        pred, proba = predict_comment(model, args.comment)
        print("\n=== Prediction ===")
        print(f"Comment: {args.comment}")
        print(f"Prediction: {LABELS[pred]} ({pred})")
        print(f"Probability [non-toxic, toxic]: [{proba[0]:.3f}, {proba[1]:.3f}]")
    elif args.demo:
        print("\n=== Demo Predictions ===")
        for s in [
            "This sucks badly.",
            "Great work team!",
            "Useless crap.",
            "Appreciate the help.",
            "Shut up already."
        ]:
            pred, proba = predict_comment(model, s)
            print(f"- {s}\n  -> Prediction: {LABELS[pred]} ({pred}) | Prob: [{proba[0]:.3f}, {proba[1]:.3f}]")
    else:
        print("\nTip: pass --comment \"your text\" to classify a new comment, or --demo for sample predictions.")

if __name__ == "__main__":
    main()
