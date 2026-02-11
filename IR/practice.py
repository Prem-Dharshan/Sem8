import math
import os
import re
from collections import Counter, defaultdict


# -------------------------------
# 1. Read Corpus
# -------------------------------

corpus_path = "./corpus"   # folder with text files

corpus = {}   # {doc_id: text}
N = 0

for filename in sorted(os.listdir(corpus_path)):

    file_path = os.path.join(corpus_path, filename)

    if os.path.isfile(file_path):

        with open(file_path, "r", errors="ignore") as f:
            text = f.read()

            N += 1
            corpus[str(N)] = text


# -------------------------------
# 2. Preprocess
# -------------------------------

stop_words = {
    "a", "the", "in", "an", "am", "that",
    "then", "i", "is", "are", "of", "and",
    "or", "not"
}

docs = defaultdict(list)   # {doc_id: [terms]}

for doc_id, text in corpus.items():

    words = text.split()

    for w in words:

        w = w.lower()                           # lowercase
        w = re.sub(r"[^\w\s]", "", w)           # remove punctuation

        if w and w not in stop_words:           # remove stopwords
            docs[doc_id].append(w)


# -------------------------------
# 3. Get Terms and Doc IDs
# -------------------------------

terms = sorted(set(
    term
    for doc_terms in docs.values()
    for term in doc_terms
))

doc_ids = sorted(docs.keys(), key=int)


# -------------------------------
# 4. Build Inverted Index + Incidence Matrix
# -------------------------------

term_incidence_matrix = defaultdict(dict)
inverted_index = defaultdict(set)


for doc_id in doc_ids:

    doc_terms = set(docs[doc_id])   # for fast lookup

    # Inverted index
    for t in doc_terms:
        inverted_index[t].add(doc_id)

    # Incidence matrix
    for t in terms:
        term_incidence_matrix[t][doc_id] = int(t in doc_terms)


# -------------------------------
# 5. Print Incidence Matrix
# -------------------------------

space = 15

print("\nTERM INCIDENCE MATRIX\n" + "=" * 60)

# Header
print("".join(
    f"{x:^{space}}"
    for x in ["Term/Doc"] + [f"D{d}" for d in doc_ids]
))

# Rows
for term in terms:

    print("".join(
        f"{x:^{space}}"
        for x in [term] + [term_incidence_matrix[term][d] for d in doc_ids]
    ))


# -------------------------------
# 6. Print Inverted Index
# -------------------------------

print("\nINVERTED INDEX\n" + "=" * 40)

for term in sorted(inverted_index):

    postings = sorted(inverted_index[term], key=int)

    print(f"{term:<15} -> {postings}")


query = "python OR file"
query_terms = query.upper().split()


def boolean_retrieval(query_terms):

    result = None
    curr_op = "AND"

    i = 0
    while i < len(query_terms):

        token = query_terms[i]

        # Operator
        if token in {"AND", "OR", "NOT"}:
            curr_op = token
            i += 1
            continue


        # Term
        term = token.lower()

        postings = set(inverted_index.get(term, []))


        # NOT
        if curr_op == "NOT":
            postings = set(doc_ids) - postings
            curr_op = "AND"

        # First term
        if result is None:
            result = postings

        # Combine
        else:
            if curr_op == "AND":
                result &= postings

            elif curr_op == "OR":
                result |= postings

        i += 1


    return result if result else set()


boolean_result = boolean_retrieval(query_terms)
print("\nBoolean Retrieval Result:", sorted(boolean_result, key=int))


document_frequency = {
    term: sum(1 for d in docs if term in docs[d])
    for term in terms
}

space = 15

print("\nDOCUMENT FREQUENCY\n" + "="*40)

print(f"{'Term':^{space}}{'DF':^{space}}")

for term in sorted(document_frequency):
    print(f"{term:^{space}}{document_frequency[term]:^{space}}")


tf = {
    doc_id: Counter(terms)
    for doc_id, terms in docs.items()
}   # {doc_id: {'python':1, 'file':1, 'data':1}}

idf = {
    term: math.log(N / document_frequency[term])
    for term in terms
}

print("\nTERM FREQUENCY (TF)\n" + "="*30)

for d in tf:
    print(f"D{d} -> {dict(tf[d])}")


print("\nINVERSE DOCUMENT FREQUENCY (IDF)\n" + "="*30)

for t in sorted(idf):
    print(f"{t:<12} -> {idf[t]:.4f}")


tfidf = {
    doc_id: {
        term: tf[doc_id][term] * idf[term]
        for term in tf[doc_id]
    }
    for doc_id in tf
}

print("\nTF-IDF\n" + "="*30)
for doc_id in sorted(tfidf, key=int):
    print(f"\nD{doc_id}:")
    for term, score in sorted(tfidf[doc_id].items()):
        print(f"  {term:<12} -> {score:.4f}")



query = "builtin python"
query_terms = query.lower().split()

# Term frequency of query
q_tf = Counter(query_terms)

# Query TF-IDF vector (dict form)
query_vector = {
    term: q_tf[term] * idf.get(term, 0)
    for term in q_tf
}

def cosine_sim(doc_vec, query_vec):

    # Dot product
    numerator = sum(
        doc_vec.get(t, 0) * query_vec.get(t, 0)
        for t in set(doc_vec) | set(query_vec)
    )

    # Magnitudes
    doc_norm = math.sqrt(sum(v*v for v in doc_vec.values()))
    query_norm = math.sqrt(sum(v*v for v in query_vec.values()))

    # Avoid division by zero
    if doc_norm == 0 or query_norm == 0:
        return 0.0

    return numerator / (doc_norm * query_norm)


def euclidean_dist(doc_vec, query_vec):

    return math.sqrt(sum(
        (doc_vec.get(t, 0) - query_vec.get(t, 0)) ** 2
        for t in set(doc_vec) | set(query_vec)
    ))

def manhattan_dist(doc_vec, query_vec):

    return sum(
        abs(doc_vec.get(t, 0) - query_vec.get(t, 0))
        for t in set(doc_vec) | set(query_vec)
    )


print("\nSIMILARITY / DISTANCE SCORES\n" + "="*40)

cos_scores = {}
euc_scores = {}
man_scores = {}

for d in tfidf:

    cos_scores[d] = cosine_sim(tfidf[d], query_vector)
    euc_scores[d] = euclidean_dist(tfidf[d], query_vector)
    man_scores[d] = manhattan_dist(tfidf[d], query_vector)


print("\nCOSINE (Higher = Better)")
for d in sorted(cos_scores, key=lambda x: cos_scores[x], reverse=True):
    print(f"D{d} -> {cos_scores[d]:.4f}")


print("\nEUCLIDEAN (Lower = Better)")
for d in sorted(euc_scores, key=lambda x: euc_scores[x]):
    print(f"D{d} -> {euc_scores[d]:.4f}")


print("\nMANHATTAN (Lower = Better)")
for d in sorted(man_scores, key=lambda x: man_scores[x]):
    print(f"D{d} -> {man_scores[d]:.4f}")


def bsm(doc_terms, query_terms):

    rsv = 0.0

    for term in query_terms:

        if term in doc_terms:

            df = document_frequency.get(term, 0)

            rsv += math.log((N - df + 0.5) / (df + 0.5))

    return rsv


query = "builtin file"
query_terms = query.lower().split()

bsm_scores = {}

for d in docs:
    bsm_scores[d] = bsm(docs[d], query_terms)


print("\nBIM / BSM SCORES\n" + "="*40)

print(f"{'Doc':^{space}}{'RSV':^{space}}")

for d in sorted(bsm_scores, key=lambda x: bsm_scores[x], reverse=True):
    print(f"D{d:^{space-1}}{bsm_scores[d]:^{space}.4f}")


average_dl = 