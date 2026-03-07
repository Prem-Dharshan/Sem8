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
from bm_preprocessing.IR import all, all_vis, eval_metrics, ndd, rel
from bm_preprocessing.DM import adaboost, apriori, bagging, hash, hunts, hunts_test, id3, id3_test, lib_doc, metrics, preprocessing
from bm_preprocessing.DM import all, all_vis

# Print the source code
print("=== IR All Module ===")
print(all)

print("\n=== DM Apriori Module ===")
print(apriori)

print("\n=== DM AdaBoost Module ===")
print(adaboost)

print("\n=== DM Bagging Module ===")
print(bagging)

print("\n=== DM Hash Module ===")
print(hash)

print("\n=== DM Hunts Module ===")
print(hunts)

print("\n=== DM Hunts Test Module ===")
print(hunts_test)

print("\n=== DM ID3 Module ===")
print(id3)

print("\n=== DM ID3 Test Module ===")
print(id3_test)

print("\n=== DM Metrics Module ===")
print(metrics)

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

>>> from bm_preprocessing.DM import adaboost
>>> print(adaboost)
# Prints entire DM/adaboost.py source code

>>> from bm_preprocessing.DM import bagging
>>> print(bagging)
# Prints entire DM/bagging.py source code

>>> from bm_preprocessing.DM import hunts, hunts_test
>>> print(hunts)
# Prints entire DM/hunts.py source code
>>> print(hunts_test)
# Prints entire DM/hunts_test.py source code

>>> from bm_preprocessing.DM import id3, id3_test
>>> print(id3)
# Prints entire DM/id3.py source code
>>> print(id3_test)
# Prints entire DM/id3_test.py source code

>>> from bm_preprocessing.DM import metrics
>>> print(metrics)
# Prints entire DM/metrics.py source code
```

---

## One-liner in Terminal

```bash
python -c "from bm_preprocessing.IR import all; print(all)"
python -c "from bm_preprocessing.IR import all_vis; print(all_vis)"
python -c "from bm_preprocessing.IR import eval_metrics; print(eval_metrics)"
python -c "from bm_preprocessing.IR import ndd; print(ndd)"
python -c "from bm_preprocessing.IR import rel; print(rel)"
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
python -c "from bm_preprocessing.DM import lib_doc; print(lib_doc)"
python -c "from bm_preprocessing.DM import preprocessing; print(preprocessing)"
```

---

## Available Modules

| Import | Description |
|--------|-------------|
| `from bm_preprocessing.IR import all` | Information Retrieval (MinHash, LSH, Rocchio, Jaccard, VS) |
| `from bm_preprocessing.IR import all_vis` | IR algorithms with Matplotlib visualizations |
| `from bm_preprocessing.IR import eval_metrics` | Jaccard, PRF, Compression Ratio, MAP metrics & plots |
| `from bm_preprocessing.IR import ndd` | Near Duplicate Documents (MinHash & LSH) |
| `from bm_preprocessing.IR import rel` | Relevance feedback & query expansion (Rocchio & LCA) |
| `from bm_preprocessing.DM import all` | All DM algorithms (Hunt's, ID3, Bagging, AdaBoost, metrics) |
| `from bm_preprocessing.DM import all_vis` | All DM algorithms + graphviz & full visualization |
| `from bm_preprocessing.DM import apriori` | Apriori algorithm |
| `from bm_preprocessing.DM import adaboost` | Bagging & AdaBoost ensemble classifiers |
| `from bm_preprocessing.DM import bagging` | Bagging ensemble classifier |
| `from bm_preprocessing.DM import hash` | Hash-based mining |
| `from bm_preprocessing.DM import hunts` | Hunt's decision tree algorithm |
| `from bm_preprocessing.DM import hunts_test` | Hunt's decision tree with visualization |
| `from bm_preprocessing.DM import id3` | ID3 decision tree algorithm |
| `from bm_preprocessing.DM import id3_test` | ID3 decision tree with visualization |
| `from bm_preprocessing.DM import metrics` | Classification metrics & curves |
| `from bm_preprocessing.DM import lib_doc` | Pandas/NumPy/Sklearn/DM/IR cheat sheet |
| `from bm_preprocessing.DM import preprocessing` | Data preprocessing utilities |



# Print cohesive modules
python -c "from bm_preprocessing.IR import all; print(all)"
python -c "from bm_preprocessing.IR import all_vis; print(all_vis)"

# Print specific algorithms
python -c "from bm_preprocessing.IR import ndd; print(ndd)"
python -c "from bm_preprocessing.IR import rel; print(rel)"
python -c "from bm_preprocessing.IR import eval_metrics; print(eval_metrics)"
