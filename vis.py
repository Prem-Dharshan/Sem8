# ================================
# IMPORTS
# ================================

import csv
from collections import defaultdict
from itertools import chain, combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ================================
# FILE INPUT & PREPROCESSING
# ================================

filename = "data.csv"  # change: data.csv / data.txt / data.xlsx


# ---------- Read File ----------

if filename.endswith(".csv") or filename.endswith(".txt"):
    df = pd.read_csv(filename)

elif filename.endswith(".xlsx"):
    df = pd.read_excel(filename)

else:
    raise ValueError("Unsupported file format")


print("Original Data:")
print(df)


# ---------- Cleaning ----------

for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

df = df.replace("", pd.NA)


# ---------- Fill Missing Values ----------

for col in df.columns:

    if df[col].dtype in ["int64", "float64"]:

        df[col] = df[col].fillna(df[col].mean())

    else:

        df[col] = df[col].fillna(df[col].mode()[0])


df = df.drop_duplicates()


print("\nAfter Cleaning:")
print(df)


# ---------- Convert to Transactions ----------

transactions = {}

for i, row in df.iterrows():

    tid = f"T{i+1}"
    items = set()

    for col in df.columns:
        item = f"{col}={row[col]}"
        items.add(item)

    transactions[tid] = items


print("\nTransactions:")
for t, v in transactions.items():
    print(t, "->", v)


# ================================
# PARAMETERS
# ================================

min_support = 2
min_conf = 0.7


# ================================
# APRIORI
# ================================


def apriori(transactions, min_support):

    C, L = {}, {}

    def count(candidates):

        count = defaultdict(int)

        for cand in candidates:
            for t in transactions.values():
                if all(i in t for i in cand):
                    count[cand] += 1

        return count

    items = sorted(set(i for t in transactions.values() for i in t))

    C[1] = count([(i,) for i in items])
    L[1] = {k: v for k, v in C[1].items() if v >= min_support}

    k = 2

    while L[k - 1]:

        candidates = set()

        for a, b in combinations(L[k - 1], 2):

            union = tuple(sorted(set(a) | set(b)))

            if len(union) == k:
                candidates.add(union)

        C[k] = count(candidates)

        L[k] = {k: v for k, v in C[k].items() if v >= min_support}

        if not L[k]:
            break

        k += 1

    return C, L


# ================================
# APRIORI + HASH (PCY)
# ================================


def apriori_hash(transactions, min_support):

    C, L = {}, {}

    C[1] = defaultdict(int)

    for t in transactions.values():
        for i in t:
            C[1][frozenset([i])] += 1

    L[1] = {k: v for k, v in C[1].items() if v >= min_support}

    h = lambda p, b=7: sum(sum(ord(c) for c in s) for s in p) % b

    buckets = defaultdict(int)

    for t in transactions.values():
        for p in combinations(sorted(t), 2):
            buckets[h(p)] += 1

    freq = {b for b, c in buckets.items() if c >= min_support}

    C[2] = defaultdict(int)

    for t in transactions.values():
        for p in combinations(sorted(t), 2):
            if h(p) in freq:
                C[2][frozenset(p)] += 1

    L[2] = {k: v for k, v in C[2].items() if v >= min_support}

    k = 3

    while L[k - 1]:

        C[k] = {
            frozenset(a | b): 0
            for a, b in combinations(L[k - 1], 2)
            if sorted(a)[: k - 2] == sorted(b)[: k - 2]
        }

        for t in transactions.values():
            for c in C[k]:
                if c.issubset(t):
                    C[k][c] += 1

        L[k] = {k: v for k, v in C[k].items() if v >= min_support}

        if not L[k]:
            break

        k += 1

    return C, L


# ================================
# ECLAT
# ================================


def eclat(transactions, min_support):

    C = {1: defaultdict(set)}

    for tid, items in transactions.items():
        for i in items:
            C[1][frozenset([i])].add(tid)

    L = {1: {k: v for k, v in C[1].items() if len(v) >= min_support}}

    k = 2

    while L[k - 1]:

        prev = list(L[k - 1].keys())

        C[k] = {
            frozenset(a | b): L[k - 1][a] & L[k - 1][b]
            for i, a in enumerate(prev)
            for b in prev[i + 1 :]
            if sorted(a)[: k - 2] == sorted(b)[: k - 2]
        }

        L[k] = {k: v for k, v in C[k].items() if len(v) >= min_support}

        if not L[k]:
            break

        k += 1

    return C, L


# ================================
# ASSOCIATION RULES
# ================================


def generate_rules(freq, total):

    rules = []

    # Convert keys to frozenset and values to counts
    freq_fixed = {}

    for k, v in freq.items():

        key = frozenset(k)

        # If value is a set (ECLAT), take length
        if isinstance(v, set):
            freq_fixed[key] = len(v)

        # Else (Apriori/PCY), already int
        else:
            freq_fixed[key] = v


    for itemset, count in freq_fixed.items():

        if len(itemset) < 2:
            continue


        for a in chain.from_iterable(
            combinations(itemset, r)
            for r in range(1, len(itemset))
        ):

            antecedent = frozenset(a)
            consequent = itemset - antecedent


            if not consequent:
                continue


            # Safe confidence calculation
            conf = count / freq_fixed[antecedent]
            sup = count / total


            if conf >= min_conf:
                rules.append((antecedent, consequent, sup, conf))


    return rules



# ================================
# RUN ALGORITHMS
# ================================

print("\n===== APRIORI =====")
C_ap, L_ap = apriori(transactions, min_support)

print("\n===== APRIORI + HASH =====")
C_pcy, L_pcy = apriori_hash(transactions, min_support)

print("\n===== ECLAT =====")
C_e, L_e = eclat(transactions, min_support)


# ================================
# FLATTEN FREQUENT ITEMSETS
# ================================


def flatten(L):

    return {k: v for Lk in L.values() for k, v in Lk.items()}


freq_ap = flatten(L_ap)
freq_pcy = flatten(L_pcy)
freq_e = flatten(L_e)


# ================================
# RULES
# ================================

total = len(transactions)

rules_ap = generate_rules(freq_ap, total)
rules_pcy = generate_rules(freq_pcy, total)
rules_e = generate_rules(freq_e, total)


# ================================
# PRINT RESULTS
# ================================


def print_L(L, name):

    print(f"\n{name}")

    for k, v in L.items():

        print(f"\nL{k}")

        for i in v:
            print(set(i))


print_L(L_ap, "APRIORI")
print_L(L_pcy, "APRIORI + HASH")
print_L(L_e, "ECLAT")


# ================================
# PRINT RULES
# ================================


def print_rules(rules, name):

    print(f"\n{name} RULES")

    for a, c, s, conf in rules:
        print(set(a), "=>", set(c), "| sup:", round(s, 2), "conf:", round(conf, 2))


print_rules(rules_ap, "APRIORI")
print_rules(rules_pcy, "PCY")
print_rules(rules_e, "ECLAT")


# ================================
# VISUALIZATION
# ================================


def plot_L(L, title):

    k_vals = []
    counts = []

    for k, v in L.items():
        k_vals.append(k)
        counts.append(len(v))

    plt.figure()
    plt.bar(k_vals, counts)
    plt.xlabel("Itemset Size (k)")
    plt.ylabel("Frequent Itemsets")
    plt.title(title)
    plt.show()


plot_L(L_ap, "Apriori Frequent Itemsets")
plot_L(L_pcy, "Apriori+Hash Frequent Itemsets")
plot_L(L_e, "ECLAT Frequent Itemsets")
