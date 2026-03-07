# bm-preprocessing Usage Guide

## Installation

```bash
pip install bm-preprocessing
```

---

## Usage in Python Scripts

You can directly import and utilize the predefined Data Mining (DM) and Information Retrieval (IR) configurations. Be sure to import from either `bm_preprocessing.DM` or `bm_preprocessing.IR` depending on the subject.

```python
# Import cohesive all-in-one modules
from bm_preprocessing.IR import all, all_vis
from bm_preprocessing.DM import all, all_vis

# Import DM-specific algorithms
from bm_preprocessing.DM import adaboost, apriori, bagging, hash, hunts, hunts_test, id3, id3_test, lib_doc, metrics, preprocessing

# Import IR-specific algorithms
from bm_preprocessing.IR import eval_metrics, ndd, rel

# Print the source code directly
print("=== IR All Module ===")
print(all)

print("\n=== IR Near Duplicate Documents ===")
print(ndd)

print("\n=== DM AdaBoost Module ===")
print(adaboost)
```

Run it locally:
```bash
python example.py
```

---

## Usage in Terminal (Interactive Python)

If you just need quick access to the source code during an exam or practical, spin up the Python REPL:

```bash
python
```

Then drop into the REPL to retrieve the code:

```python
# Returns entire IR source code cohesive module
>>> from bm_preprocessing.IR import all
>>> print(all)

# Returns Data Mining AdaBoost source code
>>> from bm_preprocessing.DM import adaboost
>>> print(adaboost)

# Returns minhash and LSH source code
>>> from bm_preprocessing.IR import ndd
>>> print(ndd)
```

---

## One-liner in Terminal

If you want the terminal to automatically print the file contents for you without entering the REPL, you can execute these one-liners directly in your Bash/PowerShell:

### Information Retrieval (IR)
```bash
python -c "from bm_preprocessing.IR import all; print(all)"
python -c "from bm_preprocessing.IR import all_vis; print(all_vis)"
python -c "from bm_preprocessing.IR import ndd; print(ndd)"
python -c "from bm_preprocessing.IR import rel; print(rel)"
python -c "from bm_preprocessing.IR import eval_metrics; print(eval_metrics)"
```

### Data Mining (DM)
```bash
python -c "from bm_preprocessing.DM import all; print(all)"
python -c "from bm_preprocessing.DM import all_vis; print(all_vis)"
python -c "from bm_preprocessing.DM import apriori; print(apriori)"
python -c "from bm_preprocessing.DM import adaboost; print(adaboost)"
python -c "from bm_preprocessing.DM import bagging; print(bagging)"
python -c "from bm_preprocessing.DM import hash; print(hash)"
python -c "from bm_preprocessing.DM import hunts; print(hunts)"
python -c "from bm_preprocessing.DM import hunts_test; print(hunts_test)"
python -c "from bm_preprocessing.DM import id3; print(id3)"
python -c "from bm_preprocessing.DM import id3_test; print(id3_test)"
python -c "from bm_preprocessing.DM import metrics; print(metrics)"
python -c "from bm_preprocessing.DM import preprocessing; print(preprocessing)"
python -c "from bm_preprocessing.DM import lib_doc; print(lib_doc)"
```

---

## Available Modules Reference

| Import Path | Description |
|-------------|-------------|
| **Information Retrieval (IR)** | |
| `from bm_preprocessing.IR import all` | Cohesive IR File: MinHash, LSH, Rocchio, Jaccard, VS |
| `from bm_preprocessing.IR import all_vis` | Cohesive IR File + Matplotlib visualizations & Heatmaps |
| `from bm_preprocessing.IR import ndd` | Near Duplicate Documents (MinHash & LSH) |
| `from bm_preprocessing.IR import rel` | Relevance feedback & query expansion (Rocchio & LCA) |
| `from bm_preprocessing.IR import eval_metrics` | Jaccard, PRF, Compression Ratios, MAP metrics & plots |
| **Data Mining (DM)** | |
| `from bm_preprocessing.DM import all` | Cohesive DM File: Hunt's, ID3, Bagging, AdaBoost, Metrics |
| `from bm_preprocessing.DM import all_vis` | Cohesive DM File + Graphviz & Matplotlib visualizations |
| `from bm_preprocessing.DM import apriori` | Apriori algorithm |
| `from bm_preprocessing.DM import adaboost` | Bagging & AdaBoost ensemble classifiers |
| `from bm_preprocessing.DM import bagging` | Bagging ensemble classifier |
| `from bm_preprocessing.DM import hash` | Hash-based mining |
| `from bm_preprocessing.DM import hunts` | Hunt's decision tree algorithm |
| `from bm_preprocessing.DM import hunts_test` | Hunt's decision tree with dataset visualization |
| `from bm_preprocessing.DM import id3` | ID3 decision tree algorithm |
| `from bm_preprocessing.DM import id3_test` | ID3 decision tree with dataset visualization |
| `from bm_preprocessing.DM import metrics` | Classification metrics & curves |
| `from bm_preprocessing.DM import preprocessing` | Data preprocessing utilities |
| `from bm_preprocessing.DM import lib_doc` | Pandas, NumPy, Sklearn cheat sheet (DM & IR logic) |
