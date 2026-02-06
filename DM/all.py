from itertools import combinations, chain
from collections import defaultdict

min_support = 2
min_conf = 0.7

transactions = {
    "T1": {"I1","I2","I4","I5","I6"},
    "T2": {"I2","I4","I6"},
    "T3": {"I2","I3"},
    "T4": {"I1","I2","I4"},
    "T5": {"I1","I2","I3"},
    "T6": {"I2","I3"},
    "T7": {"I1","I3"},
    "T8": {"I1","I2","I3","I5"},
    "T9": {"I1","I2","I3"},
    "T10": {"I1","I2","I4","I5"},
    "T11": {"I5","I6"}
}

genL = lambda C: {k: v for k, v in C.items() if v >= min_support}

C, L = {}, {}

C[1] = defaultdict(int)
for t in transactions.values():
    for i in t:
        C[1][frozenset([i])] += 1
L[1] = genL(C[1])

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
L[2] = genL(C[2])

k = 3
while L[k - 1]:
    C[k] = {
        frozenset(a | b): 0
        for a, b in combinations(L[k - 1], 2)
        if sorted(a)[: k - 2] == sorted(b)[: k - 2]
        and all(frozenset(s) in L[k - 1] for s in combinations(a | b, k - 1))
    }
    for t in transactions.values():
        for c in C[k]:
            if c.issubset(t):
                C[k][c] += 1
    L[k] = genL(C[k])
    if not L[k]:
        break
    k += 1

frequent_itemsets = {k: v for Lk in L.values() for k, v in Lk.items()}
total = len(transactions)
rules = []
for itemset, count in frequent_itemsets.items():
    if len(itemset) < 2:
        continue
    for a in chain.from_iterable(
        combinations(itemset, r) for r in range(1, len(itemset))
    ):
        antecedent = frozenset(a)
        consequent = itemset - antecedent
        if not consequent:
            continue
        support = count / total
        confidence = (
            count / C[1 if len(antecedent) == 1 else len(antecedent)][antecedent]
        )
        if confidence >= min_conf:
            rules.append((antecedent, consequent, support, confidence))

for k, v in C.items():
    print(
        f"\nC{k}:\n",
        "Empty" if not v else "\n".join(f"{set(x)} : {y}" for x, y in v.items()),
    )
for k, v in L.items():
    print(f"\nL{k}:\n", "Empty" if not v else "\n".join(f"{set(x)} : {v[x]}" for x in v))

print(f"\nAssociation Rules (conf >= {min_conf:.0%}):")
for a, c, s, conf in rules:
    print(f"{set(a)} => {set(c)} | support: {s:.2f}, confidence: {conf:.2f}")

from itertools import combinations, chain

transactions = {
    "10": {"A", "C", "D"},
    "20": {"B", "C", "E"},
    "30": {"A", "B", "C", "E"},
    "40": {"B", "E"},
}

min_support = 2
min_conf = 0.7

genL = lambda C: {k: v for k, v in C.items() if len(v) >= min_support}

C = {1: {}}
for tid, items in transactions.items():
    for i in items:
        C[1].setdefault(frozenset([i]), set()).add(tid)

L = {1: genL(C[1])}

k = 2
while L[k - 1]:
    prev = list(L[k - 1].keys())
    C[k] = {
        frozenset(a | b): L[k - 1][a] & L[k - 1][b]
        for i, a in enumerate(prev)
        for b in prev[i + 1 :]
        if sorted(a)[: k - 2] == sorted(b)[: k - 2]
    }
    L[k] = genL(C[k])
    if not L[k]:
        break
    k += 1

frequent_itemsets = {k: v for Lk in L.values() for k, v in Lk.items()}
total = len(transactions)
rules = []
for itemset, tids in frequent_itemsets.items():
    if len(itemset) < 2:
        continue
    for a in chain.from_iterable(
        combinations(itemset, r) for r in range(1, len(itemset))
    ):
        antecedent = frozenset(a)
        consequent = itemset - antecedent
        if len(consequent) == 0:
            continue
        support = len(tids) / total
        confidence = len(tids) / len(frequent_itemsets[antecedent])
        if confidence >= min_conf:
            rules.append((antecedent, consequent, support, confidence))

for k, v in C.items():
    print(f"\nC{k}:")
    print("Empty" if not v else "\n".join(f"{set(x)} : {y}" for x, y in v.items()))

for k, v in L.items():
    print(f"\nL{k}:")
    print("Empty" if not v else "\n".join(f"{set(x)} : {len(y)}" for x, y in v.items()))

print(f"\nAssociation Rules (conf >= {min_conf:.0%}):")
for a, c, s, conf in rules:
    print(f"{set(a)} => {set(c)} | support: {s:.2f}, confidence: {conf:.2f}")
