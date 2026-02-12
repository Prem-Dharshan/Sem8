import math
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

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

# ---------- Corpus ----------
docs = [
    "information retrieval is fun",
    "retrieval models are boolean vector probabilistic",
    "information theory and probability",
    "boolean retrieval is simple"
]

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

def tfidf_func(doc):
    return {t: tf(doc)[t] * idf(t) for t in doc}

doc_vectors = [tfidf_func(doc) for doc in processed_docs]
query_vector = tfidf_func(preprocess("information retrieval"))

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

# ---------- 8. Okapi BM25 ----------
avg_dl = sum(len(doc) for doc in processed_docs) / N
k1, b = 1.5, 0.75

def bm25(doc, query_terms):
    score = 0.0
    doc_len = len(doc)
    freqs = Counter(doc)

    for term in query_terms:
        if term in freqs:
            df = sum(1 for d in processed_docs if term in d)
            idf = math.log((N - df + 0.5) / (df + 0.5))
            tf = freqs[term]
            score += idf * ((tf * (k1 + 1)) /
                     (tf + k1 * (1 - b + b * doc_len / avg_dl)))
    return score

bm25_scores = {
    i: bm25(processed_docs[i], preprocess("information retrieval"))
    for i in range(N)
}

print("\nBM25 Scores:", bm25_scores)

# ---------- 7. Evaluation Metrics ----------
relevant_docs = {0, 3}  # ground truth

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

# ---------- 9. Compare Models ----------
print("\nEvaluation Metrics:")
print("Boolean:", evaluate(boolean_result))
print("VSM:", evaluate([i for i, s in vsm_scores.items() if s > 0]))
print("BIM:", evaluate([i for i, s in bim_scores.items() if s > 0]))
print("BM25:", evaluate([i for i, s in bm25_scores.items() if s > 0]))


import matplotlib.pyplot as plt


models = ["Boolean", "VSM", "BIM", "BM25"]

bool_eval = evaluate(boolean_result)
vsm_eval = evaluate([i for i, s in vsm_scores.items() if s > 0])
bim_eval = evaluate([i for i, s in bim_scores.items() if s > 0])
bm25_eval = evaluate([i for i, s in bm25_scores.items() if s > 0])

evaluations = [
    bool_eval,
    vsm_eval,
    bim_eval,
    bm25_eval
]

accuracy  = [e[0] for e in evaluations]
precision = [e[1] for e in evaluations]
recall    = [e[2] for e in evaluations]
f1        = [e[3] for e in evaluations]


x = range(len(models))
width = 0.2

plt.figure()

plt.bar(x, accuracy,  width=width, label="Accuracy")
plt.bar([i+width for i in x], precision, width=width, label="Precision")
plt.bar([i+2*width for i in x], recall,    width=width, label="Recall")
plt.bar([i+3*width for i in x], f1,        width=width, label="F1")

plt.xticks([i+1.5*width for i in x], models)
plt.ylabel("Score")
plt.title("Evaluation Metrics Comparison")
plt.legend()

plt.show()

plt.figure()

plt.plot(vsm_scores.keys(), vsm_scores.values(), marker='o', label="VSM")
plt.plot(bim_scores.keys(), bim_scores.values(), marker='o', label="BIM")
plt.plot(bm25_scores.keys(), bm25_scores.values(), marker='o', label="BM25")

plt.xlabel("Document ID")
plt.ylabel("Score")
plt.title("Document Ranking Scores")
plt.legend()

plt.show()

def confusion_matrix(retrieved, relevant, N):

    retrieved = set(retrieved)

    tp = len(retrieved & relevant)
    fp = len(retrieved - relevant)
    fn = len(relevant - retrieved)
    tn = N - tp - fp - fn

    return [[tp, fp],
            [fn, tn]]

bm25_retrieved = [i for i, s in bm25_scores.items() if s > 0]

cm = confusion_matrix(bm25_retrieved, relevant_docs, N)

plt.figure()

plt.imshow(cm)
plt.colorbar()

plt.xticks([0,1], ["Relevant", "Not Relevant"])
plt.yticks([0,1], ["Retrieved", "Not Retrieved"])

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("BM25 Confusion Matrix")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i][j], ha="center", va="center")

plt.show()

def precision_recall_curve(scores, relevant):

    ranked = sorted(scores.items(),
                    key=lambda x: x[1],
                    reverse=True)

    tp = 0
    fp = 0

    precisions = []
    recalls = []

    for doc, _ in ranked:

        if doc in relevant:
            tp += 1
        else:
            fp += 1

        precision = tp / (tp + fp)
        recall = tp / len(relevant)

        precisions.append(precision)
        recalls.append(recall)

    return recalls, precisions

rec, prec = precision_recall_curve(bm25_scores, relevant_docs)

plt.figure()

plt.plot(rec, prec, marker="o")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (BM25)")

plt.grid()

plt.show()


tfidf_docs = {
    str(i+1): doc_vectors[i]
    for i in range(N)
}

tfidf_matrix = [
    [tfidf_docs[str(d+1)].get(t, 0) for t in terms]
    for d in range(N)
]


plt.figure()

plt.imshow(tfidf_matrix)
plt.colorbar()

plt.xlabel("Terms")
plt.ylabel("Documents")
plt.title("TF-IDF Heatmap")

plt.show()



def rank_scores(scores_dict):
    return sorted(scores_dict.items(), key= lambda x: x[1], reverse=True)


def plot_scores(scores, title):

    ranked = rank_scores(scores)

    docs = [i for i, _ in ranked]
    vals = [s for _, s in ranked]

    plt.figure(figsize=(8, 6))
    plt.bar(docs, vals)



def plot_metrics():

    models = ["Boolean", "VSM", "BIM", "BM25", "Naive Bayes"]

    results = {
        "Boolean": evaluate(boolean_result),
        "VSM": evaluate([i for i, s in vsm_scores.items() if s > 0]),
        "BIM": evaluate([i for i, s in bim_scores.items() if s > 0]),
        "BM25": evaluate([i for i, s in bm25_scores.items() if s > 0]),
        "Naive Bayes" : evaluate([i for i, s in nb_scores.items() if s > -999999]),
    }

    accuracy = [results[m][0] for m in models]
    precision = [results[m][1] for m in models]
    recall = [results[m][2] for m in models]
    f1 = [results[m][3] for m in models]

    plt.figure(figsize=(8, 6))

    