import math
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

import pandas as pd
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


# ---------- 1. Corpus Input (Hardcoded OR CSV) ----------
# OPTION A: Hardcoded documents (your original)
docs = [
    "information retrieval is fun",
    "retrieval models are boolean vector probabilistic",
    "information theory and probability",
    "boolean retrieval is simple"
]

# OPTION B: Load from CSV files (1 or more)
# CSV must contain a column: text (or change below)
# csv_files = ["file1.csv", "file2.csv"]

csv_files = []  # put file names here if needed

def load_docs_from_csv(csv_files, text_column="text"):
    all_docs = []
    for file in csv_files:
        df = pd.read_csv(file)
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in {file}. Columns: {df.columns}")
        all_docs.extend(df[text_column].dropna().astype(str).tolist())
    return all_docs

if csv_files:
    docs = load_docs_from_csv(csv_files, text_column="text")

processed_docs = [preprocess(doc) for doc in docs]
N = len(docs)


# ---------- 2. Term Incidence Matrix ----------
terms = sorted(set(term for doc in processed_docs for term in doc))

term_incidence = {
    term: [1 if term in doc else 0 for doc in processed_docs]
    for term in terms
}

print("\nTerm Incidence Matrix:")
for term, row in term_incidence.items():
    print(term, row)


# ---------- 3. Inverted Index ----------
inverted_index = defaultdict(list)

for doc_id, doc in enumerate(processed_docs):
    for term in set(doc):
        inverted_index[term].append(doc_id)

print("\nInverted Index:")
for term, postings in inverted_index.items():
    print(term, postings)


# ---------- Query ----------
query = "information AND NOT boolean"
query_terms = preprocess(query)


# ---------- 4. Boolean Model (AND / OR / NOT) ----------
def boolean_retrieval(query):
    tokens = query.upper().split()
    result = set()
    current_op = None

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token in {"AND", "OR", "NOT"}:
            current_op = token
        else:
            term = preprocess(token.lower())
            postings = set()

            if term and term[0] in inverted_index:
                postings = set(inverted_index[term[0]])

            if current_op == "NOT":
                postings = set(range(N)) - postings
                current_op = None

            if not result:
                result = postings
            else:
                if current_op == "AND":
                    result = result & postings
                elif current_op == "OR":
                    result = result | postings

        i += 1

    return result

boolean_result = boolean_retrieval(query)
print("\nBoolean Retrieval Result:", boolean_result)


# ---------- 5. Vector Space Model (TF-IDF) ----------
def tf(doc):
    return Counter(doc)

def idf(term):
    df = sum(1 for d in processed_docs if term in d)
    return math.log(N / (df + 1))

def tfidf(doc):
    return {t: tf(doc)[t] * idf(t) for t in doc}

doc_vectors = [tfidf(doc) for doc in processed_docs]
query_vector = tfidf(preprocess("information retrieval"))

def cosine_similarity(v1, v2):
    num = sum(v1.get(t, 0) * v2.get(t, 0) for t in set(v1) | set(v2))
    den1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    den2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    return num / (den1 * den2) if den1 and den2 else 0

vsm_scores = {
    i: cosine_similarity(query_vector, doc_vectors[i])
    for i in range(N)
}

print("\nVector Space Model Scores:", vsm_scores)


# ---------- 6. Probabilistic Model (BIM with RSV) ----------
def bim_rsv(doc, query_terms):
    rsv = 0.0
    for term in query_terms:
        if term in doc:
            df = sum(1 for d in processed_docs if term in d)
            rsv += math.log((N - df + 0.5) / (df + 0.5))
    return rsv

bim_scores = {
    i: bim_rsv(processed_docs[i], preprocess("information retrieval"))
    for i in range(N)
}

print("\nBIM RSV Scores:", bim_scores)


# ---------- 7. Okapi BM25 ----------
avg_dl = sum(len(doc) for doc in processed_docs) / N
k1, b = 1.5, 0.75

def bm25(doc, query_terms):
    score = 0.0
    doc_len = len(doc)
    freqs = Counter(doc)

    for term in query_terms:
        if term in freqs:
            df = sum(1 for d in processed_docs if term in d)
            idf_val = math.log((N - df + 0.5) / (df + 0.5))
            tf_val = freqs[term]
            score += idf_val * ((tf_val * (k1 + 1)) /
                     (tf_val + k1 * (1 - b + b * doc_len / avg_dl)))
    return score

bm25_scores = {
    i: bm25(processed_docs[i], preprocess("information retrieval"))
    for i in range(N)
}

print("\nBM25 Scores:", bm25_scores)


# ---------- 8. Naive Bayes (Multinomial) ----------
# This is a simple NB ranking based on P(doc | query) ~ P(query | doc)
# Using Laplace smoothing

vocab = sorted(set(term for doc in processed_docs for term in doc))
V = len(vocab)

doc_term_freqs = [Counter(doc) for doc in processed_docs]
doc_lengths = [len(doc) for doc in processed_docs]

def naive_bayes_score(doc_id, query_terms):
    score = 0.0
    freqs = doc_term_freqs[doc_id]
    dl = doc_lengths[doc_id]

    for term in query_terms:
        term_count = freqs.get(term, 0)
        prob = (term_count + 1) / (dl + V)  # Laplace smoothing
        score += math.log(prob)

    return score

nb_scores = {
    i: naive_bayes_score(i, preprocess("information retrieval"))
    for i in range(N)
}

print("\nNaive Bayes Scores:", nb_scores)


# ---------- 9. Evaluation Metrics ----------
relevant_docs = {0, 3}  # ground truth (example)

def evaluate(retrieved):
    retrieved = set(retrieved)
    tp = len(retrieved & relevant_docs)
    fp = len(retrieved - relevant_docs)
    fn = len(relevant_docs - retrieved)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    accuracy = tp / N

    return accuracy, precision, recall, f1


# ---------- 10. Compare Models ----------
print("\nEvaluation Metrics:")
print("Boolean:", evaluate(boolean_result))
print("VSM:", evaluate([i for i, s in vsm_scores.items() if s > 0]))
print("BIM:", evaluate([i for i, s in bim_scores.items() if s > 0]))
print("BM25:", evaluate([i for i, s in bm25_scores.items() if s > 0]))
print("Naive Bayes:", evaluate([i for i, s in nb_scores.items() if s > -999999]))


# ---------- 11. Ranking Output ----------
def rank_scores(score_dict):
    return sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

print("\nRanking Results:")
print("VSM Ranking:", rank_scores(vsm_scores))
print("BIM Ranking:", rank_scores(bim_scores))
print("BM25 Ranking:", rank_scores(bm25_scores))
print("Naive Bayes Ranking:", rank_scores(nb_scores))


# ---------- 12. Visualization ----------
def plot_scores(scores, title):
    ranked = rank_scores(scores)
    doc_ids = [f"D{i}" for i, _ in ranked]
    values = [s for _, s in ranked]

    plt.figure(figsize=(8, 4))
    plt.bar(doc_ids, values)
    plt.title(title)
    plt.xlabel("Documents")
    plt.ylabel("Score")
    plt.show()

plot_scores(vsm_scores, "Vector Space Model (TF-IDF Cosine) Scores")
plot_scores(bim_scores, "Probabilistic Model (BIM RSV) Scores")
plot_scores(bm25_scores, "BM25 Scores")
plot_scores(nb_scores, "Naive Bayes Scores")


def plot_metrics():
    models = ["Boolean", "VSM", "BIM", "BM25", "Naive Bayes"]

    results = {
        "Boolean": evaluate(boolean_result),
        "VSM": evaluate([i for i, s in vsm_scores.items() if s > 0]),
        "BIM": evaluate([i for i, s in bim_scores.items() if s > 0]),
        "BM25": evaluate([i for i, s in bm25_scores.items() if s > 0]),
        "Naive Bayes": evaluate([i for i, s in nb_scores.items() if s > -999999]),
    }

    accuracy = [results[m][0] for m in models]
    precision = [results[m][1] for m in models]
    recall = [results[m][2] for m in models]
    f1 = [results[m][3] for m in models]

    plt.figure(figsize=(10, 4))
    plt.bar(models, accuracy)
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.bar(models, precision)
    plt.title("Precision Comparison")
    plt.ylabel("Precision")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.bar(models, recall)
    plt.title("Recall Comparison")
    plt.ylabel("Recall")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.bar(models, f1)
    plt.title("F1 Score Comparison")
    plt.ylabel("F1 Score")
    plt.show()

plot_metrics()
