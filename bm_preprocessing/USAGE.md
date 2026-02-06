# bm-preprocessing Usage Guide

## Installation

```bash
pip install bm-preprocessing
```

---

## Usage in Python File

Create a file `example.py`:

```python
# Import modules
from bm_preprocessing.IR import all
from bm_preprocessing.DM import apriori, hash, preprocessing

# Print the source code
print("=== IR All Module ===")
print(all)

print("\n=== DM Apriori Module ===")
print(apriori)

print("\n=== DM Hash Module ===")
print(hash)

print("\n=== DM Preprocessing Module ===")
print(preprocessing)
```

Run it:
```bash
python example.py
```

---

## Usage in Terminal (Interactive Python)

```bash
python
```

Then in the Python REPL:

```python
>>> from bm_preprocessing.IR import all
>>> print(all)
# Prints entire IR/all.py source code

>>> from bm_preprocessing.DM import apriori
>>> print(apriori)
# Prints entire DM/apriori.py source code
```

---

## One-liner in Terminal

```bash
python -c "from bm_preprocessing.IR import all; print(all)"
python -c "from bm_preprocessing.DM import apriori; print(apriori)"
python -c "from bm_preprocessing.DM import hash; print(hash)"
python -c "from bm_preprocessing.DM import preprocessing; print(preprocessing)"
```

---

## Available Modules

| Import | Description |
|--------|-------------|
| `from bm_preprocessing.IR import all` | Information Retrieval (BM25, TF-IDF, Boolean) |
| `from bm_preprocessing.DM import all` | Data Mining algorithms |
| `from bm_preprocessing.DM import apriori` | Apriori algorithm |
| `from bm_preprocessing.DM import hash` | Hash-based mining |
| `from bm_preprocessing.DM import preprocessing` | Data preprocessing utilities |
