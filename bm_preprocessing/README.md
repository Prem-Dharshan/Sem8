# 📄 `bm-preprocessing`

**`bm-preprocessing`** is a Python package providing easy-to-use NLP preprocessing utilities built on top of **NLTK** and **pandas**. It helps you clean, normalize, tokenize, and vectorize text data efficiently using a modular pipeline.

---

## ✨ Features

* Text cleaning and normalization
* Tokenization and stopword removal
* Lemmatization
* TF-IDF and Bag-of-Words vectorization
* Pipeline-based preprocessing
* Built on NLTK and pandas
* Scikit-learn–style API

---

## 📦 Installation

Install from PyPI:

```bash
pip install bm-preprocessing
```

---

## 🚀 Quick Start

### Basic Usage with Pipeline

```python
from bm_preprocessing import (
    TextCleaner,
    Tokenizer,
    Normalizer,
    StopwordFilter,
    Lemmatizer,
    Vectorizer,
    Pipeline
)

# Sample documents
documents = [
    "This is an example document! It has punctuation & numbers: 123.",
    "Natural Language Processing is AMAZING!!!",
    "Preprocessing text is very important for NLP tasks."
]

# Create preprocessing components
cleaner = TextCleaner(
    lowercase=True,
    remove_punctuation=True,
    remove_numbers=True,
    strip_whitespace=True
)

tokenizer = Tokenizer(method="word")

normalizer = Normalizer(
    expand_contractions=True,
    fix_unicode=True
)

stopword_filter = StopwordFilter(language="english")

lemmatizer = Lemmatizer(method="wordnet")

vectorizer = Vectorizer(
    method="tfidf",
    max_features=5000,
    ngram_range=(1, 2)
)

# Build pipeline
preprocessing_pipeline = Pipeline([
    cleaner,
    normalizer,
    tokenizer,
    stopword_filter,
    lemmatizer,
    vectorizer
])

# Run preprocessing
processed_data = preprocessing_pipeline.fit_transform(documents)

# Inspect output
print("Processed Features Shape:", processed_data.shape)
print("Sample Vector:", processed_data[0])
```

---

## 🧩 Step-by-Step Processing (Without Pipeline)

You can also run each step manually:

```python
from bm_preprocessing import (
    TextCleaner,
    Tokenizer,
    StopwordFilter,
    Lemmatizer,
    Vectorizer
)

docs = [
    "Machine learning is fun!",
    "Text preprocessing improves results."
]

# Initialize tools
cleaner = TextCleaner(lowercase=True)
tokenizer = Tokenizer()
stopwords = StopwordFilter("english")
lemmatizer = Lemmatizer()
vectorizer = Vectorizer(method="bow")

# Process
cleaned = [cleaner.clean(d) for d in docs]
tokens = [tokenizer.tokenize(d) for d in cleaned]
filtered = [stopwords.remove(t) for t in tokens]
lemmatized = [lemmatizer.lemmatize(t) for t in filtered]

vectors = vectorizer.fit_transform(lemmatized)

print(vectors)
```

---

## 🛠️ Components Overview

| Component        | Description                       |
| ---------------- | --------------------------------- |
| `TextCleaner`    | Removes noise and formats text    |
| `Tokenizer`      | Splits text into tokens           |
| `Normalizer`     | Standardizes text                 |
| `StopwordFilter` | Removes common filler words       |
| `Lemmatizer`     | Converts words to base form       |
| `Vectorizer`     | Converts text to numeric features |
| `Pipeline`       | Chains components into a workflow |

---

## 🧠 Deep Learning Preparation Example

For sequence models:

```python
from bm_preprocessing import (
    TextCleaner,
    Tokenizer,
    SequencePadder,
    VocabularyBuilder
)

texts = [
    "Deep learning for NLP",
    "Transformers are powerful"
]

cleaner = TextCleaner(lowercase=True)
tokenizer = Tokenizer()
vocab = VocabularyBuilder(max_size=10000)
padder = SequencePadder(max_length=50)

# Clean
cleaned = [cleaner.clean(t) for t in texts]

# Tokenize
tokens = [tokenizer.tokenize(t) for t in cleaned]

# Build vocabulary
vocab.fit(tokens)

# Encode
encoded = [vocab.encode(t) for t in tokens]

# Pad
padded = padder.pad(encoded)

print(padded)
```

---

## 📚 Requirements

* Python 3.8+
* nltk
* pandas
* scikit-learn (for vectorization)

Install dependencies automatically with:

```bash
pip install bm-preprocessing
```

---

## 📂 Project Structure

```
bm_preprocessing/
│
├── cleaning.py
├── tokenization.py
├── normalization.py
├── filtering.py
├── lemmatization.py
├── vectorization.py
├── pipeline.py
└── __init__.py
```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a pull request

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📬 Support

If you encounter any issues or have feature requests, please open an issue on GitHub.

---
