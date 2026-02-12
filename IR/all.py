import math
from collections import defaultdict, Counter
from matplotlib import pyplot as plt
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


def confusion_matrix(retrieved, relevant, N):

    retrieved = set(retrieved)

    tp = len(retrieved & relevant)
    fp = len(retrieved - relevant)
    fn = len(relevant - retrieved)
    tn = N - tp - fp - fn

    return [[tp, fp],
            [fn, tn]]


def plot_cm(scores, relevant, N, title):

    retrieved = [i for i, s in scores.items() if s > 0]

    cm = confusion_matrix(retrieved, relevant)

    plt.imshow(cm)
    plt.colorbar()

    plt.xticks([0, 1], ["rel", "n rel"])
    plt.xticks([0, 1], ["ret", "n ret"])

    
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)


    for i in [0, 1]:
        for j in [0, 1]:
            plt.text(j, i, cm[i][j], ha="center", va="center")