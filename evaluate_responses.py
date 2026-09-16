import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# FILES
# ============================================================

GOLD_FILE = "amazon_gold_200_labelled.csv"
SUPPORT_FILE = "amazon_support.csv"
OUTPUT_FILE = "response_evaluation.csv"


# ============================================================
# INTENTS
# ============================================================

INTENTS = [
    "Order & Delivery",
    "Returns & Refunds",
    "Payment & Amazon Pay",
    "Account & Security",
    "Product Problems",
    "Prime Membership & Benefits",
    "App / Device / Technical",
    "Prime Video / Streaming",
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove Twitter handles
    text = re.sub(r"@\w+", " ", text)

    # Remove reply signatures such as ^AB
    text = re.sub(r"\^\w+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


# ============================================================
# LOAD GOLD DATA
# ============================================================

print("Loading gold data...")

gold = pd.read_csv(GOLD_FILE)

gold["gold_intent"] = gold["gold_intent"].astype(str).str.strip()
gold["gold_escalate"] = (
    gold["gold_escalate"]
    .astype(str)
    .str.strip()
    .str.upper()
)

gold = gold[
    gold["gold_intent"].isin(INTENTS)
].copy()

print("Gold examples:", len(gold))


# ============================================================
# CREATE 80/20 STRATIFIED SPLIT
# ============================================================

print("\nCreating 80/20 stratified test split...")

train_df, test_df = train_test_split(
    gold,
    test_size=0.20,
    random_state=42,
    stratify=gold["gold_intent"],
)

train_df = train_df.copy()
test_df = test_df.copy()

print("Training examples:", len(train_df))
print("Test examples:", len(test_df))


# ============================================================
# INTENT CLASSIFIER
# ============================================================

print("\nTraining intent classifier on TRAINING data only...")

intent_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
)

X_train_intent = intent_vectorizer.fit_transform(
    train_df["text"].fillna("").apply(clean_text)
)

X_test_intent = intent_vectorizer.transform(
    test_df["text"].fillna("").apply(clean_text)
)

intent_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
)

intent_model.fit(
    X_train_intent,
    train_df["gold_intent"],
)

predicted_intents = intent_model.predict(
    X_test_intent
)

intent_accuracy = accuracy_score(
    test_df["gold_intent"],
    predicted_intents,
)

intent_macro_f1 = f1_score(
    test_df["gold_intent"],
    predicted_intents,
    average="macro",
    zero_division=0,
)

print("\nIntent test performance")
print("--------------------------------------------")
print("Accuracy:", round(intent_accuracy, 3))
print("Macro-F1:", round(intent_macro_f1, 3))

print("\nClassification report:")
print(
    classification_report(
        test_df["gold_intent"],
        predicted_intents,
        labels=INTENTS,
        zero_division=0,
    )
)


# ============================================================
# ESCALATION CLASSIFIER
# ============================================================

print("\nTraining escalation classifier on TRAINING data only...")

train_df["escalate_binary"] = train_df["gold_escalate"].map({
    "YES": 1,
    "NO": 0,
})

test_df["escalate_binary"] = test_df["gold_escalate"].map({
    "YES": 1,
    "NO": 0,
})

escalation_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
)

X_train_esc = escalation_vectorizer.fit_transform(
    train_df["text"].fillna("").apply(clean_text)
)

X_test_esc = escalation_vectorizer.transform(
    test_df["text"].fillna("").apply(clean_text)
)

escalation_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
)

escalation_model.fit(
    X_train_esc,
    train_df["escalate_binary"],
)

predicted_escalation_binary = escalation_model.predict(
    X_test_esc
)

predicted_escalation = np.where(
    predicted_escalation_binary == 1,
    "YES",
    "NO",
)

gold_escalation_binary = test_df["escalate_binary"].values

escalation_accuracy = accuracy_score(
    gold_escalation_binary,
    predicted_escalation_binary,
)

escalation_precision = precision_score(
    gold_escalation_binary,
    predicted_escalation_binary,
    zero_division=0,
)

escalation_recall = recall_score(
    gold_escalation_binary,
    predicted_escalation_binary,
    zero_division=0,
)

escalation_f1 = f1_score(
    gold_escalation_binary,
    predicted_escalation_binary,
    zero_division=0,
)

print("\nEscalation test performance")
print("--------------------------------------------")
print("Accuracy:", round(escalation_accuracy, 3))
print("Precision:", round(escalation_precision, 3))
print("Recall:", round(escalation_recall, 3))
print("F1:", round(escalation_f1, 3))


# ============================================================
# LOAD AMAZON SUPPORT DATA
# ============================================================

print("\nLoading Amazon support data...")

support = pd.read_csv(SUPPORT_FILE)

print("Support rows:", len(support))

support["tweet_id"] = support["tweet_id"].astype(str)


# ============================================================
# RECONSTRUCT CUSTOMER -> AMAZONHELP PAIRS
# ============================================================

print("\nBuilding customer-reply pairs...")

amazonhelp = support[
    support["author_id"]
    .astype(str)
    .str.strip()
    .str.lower()
    == "amazonhelp"
].copy()

amazon_ids = set(
    amazonhelp["tweet_id"].astype(str)
)

amazon_reply_lookup = {}

for _, row in amazonhelp.iterrows():
    amazon_reply_lookup[
        str(row["tweet_id"])
    ] = row["text"]


def contains_amazon_reply(value):

    if pd.isna(value):
        return False

    ids = str(value).split(",")

    return any(
        tweet_id.strip() in amazon_ids
        for tweet_id in ids
    )


customers = support[
    support["response_tweet_id"]
    .apply(contains_amazon_reply)
].copy()


pairs = []

for _, row in customers.iterrows():

    response_ids = str(
        row["response_tweet_id"]
    ).split(",")

    replies = []

    for response_id in response_ids:

        response_id = response_id.strip()

        if response_id in amazon_reply_lookup:
            replies.append(
                amazon_reply_lookup[response_id]
            )

    if not replies:
        continue

    pairs.append({
        "customer_text": row["text"],
        "reply_text": " ".join(replies),
    })


pairs_df = pd.DataFrame(pairs)

pairs_df["customer_clean"] = (
    pairs_df["customer_text"]
    .apply(clean_text)
)

pairs_df["reply_clean"] = (
    pairs_df["reply_text"]
    .apply(clean_text)
)

pairs_df = pairs_df[
    (pairs_df["customer_clean"] != "") &
    (pairs_df["reply_clean"] != "")
].copy()

pairs_df = pairs_df.drop_duplicates(
    subset=["customer_clean", "reply_clean"]
).reset_index(drop=True)

print("Customer-reply pairs:", len(pairs_df))


# ============================================================
# RETRIEVAL MODEL
# ============================================================

print("\nTraining retrieval TF-IDF...")

retrieval_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

retrieval_matrix = retrieval_vectorizer.fit_transform(
    pairs_df["customer_clean"]
)

print(
    "Retrieval TF-IDF shape:",
    retrieval_matrix.shape
)


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_response(query, top_k=1):

    query_clean = clean_text(query)

    query_vector = retrieval_vectorizer.transform(
        [query_clean]
    )

    similarities = cosine_similarity(
        query_vector,
        retrieval_matrix
    )[0]

    # IMPORTANT:
    # Exclude exact copies of the test query.
    # Otherwise the evaluator may retrieve the gold
    # conversation itself and create leakage.

    exact_matches = (
        pairs_df["customer_clean"]
        == query_clean
    )

    similarities[exact_matches.values] = -1

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for idx in top_indices:

        if similarities[idx] < 0:
            continue

        results.append({
            "customer": pairs_df.iloc[idx][
                "customer_text"
            ],
            "reply": pairs_df.iloc[idx][
                "reply_text"
            ],
            "similarity": float(
                similarities[idx]
            ),
        })

    return results


# ============================================================
# RESPONSE GENERATION
# ============================================================

def generate_response(
    query,
    predicted_intent,
    predicted_escalate,
):

    # Human escalation
    if predicted_escalate == "YES":

        response = (
            "I'm sorry you're experiencing this. "
            "I'd recommend contacting Amazon customer "
            "support so a specialist can look into "
            "this for you."
        )

        return response, None

    # Historical response retrieval
    retrieved = retrieve_response(
        query,
        top_k=1,
    )

    if not retrieved:

        return (
            "I'm sorry, but I couldn't find a "
            "suitable historical response for "
            "this issue."
        ), None

    return (
        retrieved[0]["reply"],
        retrieved[0],
    )


# ============================================================
# RUN FULL AGENT ON HELD-OUT TEST SET
# ============================================================

print("\nRunning full agent on HELD-OUT test set...")

results = []

for i, (_, row) in enumerate(
    test_df.iterrows(),
    start=1,
):

    query = row["text"]

    query_clean = clean_text(query)

    # ----------------------------
    # Intent
    # ----------------------------

    intent_vector = intent_vectorizer.transform(
        [query_clean]
    )

    predicted_intent = intent_model.predict(
        intent_vector
    )[0]

    # ----------------------------
    # Escalation
    # ----------------------------

    escalation_vector = (
        escalation_vectorizer.transform(
            [query_clean]
        )
    )

    escalation_prediction = (
        escalation_model.predict(
            escalation_vector
        )[0]
    )

    predicted_escalate = (
        "YES"
        if escalation_prediction == 1
        else "NO"
    )

    # ----------------------------
    # Response
    # ----------------------------

    final_response, retrieval = (
        generate_response(
            query,
            predicted_intent,
            predicted_escalate,
        )
    )

    # ----------------------------
    # Retrieval details
    # ----------------------------

    if retrieval:

        retrieved_customer = (
            retrieval["customer"]
        )

        retrieved_reply = (
            retrieval["reply"]
        )

        retrieval_similarity = (
            retrieval["similarity"]
        )

    else:

        retrieved_customer = ""
        retrieved_reply = ""
        retrieval_similarity = np.nan

    # ----------------------------
    # Save result
    # ----------------------------

    results.append({

        "tweet_id":
            row["tweet_id"],

        "text":
            query,

        "gold_intent":
            row["gold_intent"],

        "predicted_intent":
            predicted_intent,

        "intent_correct":
            predicted_intent
            == row["gold_intent"],

        "gold_escalate":
            row["gold_escalate"],

        "predicted_escalate":
            predicted_escalate,

        "escalation_correct":
            predicted_escalate
            == row["gold_escalate"],

        "retrieved_customer":
            retrieved_customer,

        "retrieved_reply":
            retrieved_reply,

        "retrieval_similarity":
            retrieval_similarity,

        "final_response":
            final_response,
    })

    if i % 10 == 0:
        print(
            f"Processed {i}/{len(test_df)} test examples"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# FALSE NEGATIVE ANALYSIS
# ============================================================

false_negatives = results_df[
    (results_df["gold_escalate"] == "YES") &
    (results_df["predicted_escalate"] == "NO")
].copy()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("============================================")
print("FINAL RESPONSE EVALUATION")
print("============================================")

print(
    "Training examples:",
    len(train_df)
)

print(
    "Held-out test examples:",
    len(test_df)
)

print(
    "\nIntent Accuracy:",
    round(intent_accuracy, 3)
)

print(
    "Intent Macro-F1:",
    round(intent_macro_f1, 3)
)

print(
    "\nEscalation Accuracy:",
    round(escalation_accuracy, 3)
)

print(
    "Escalation Precision:",
    round(escalation_precision, 3)
)

print(
    "Escalation Recall:",
    round(escalation_recall, 3)
)

print(
    "Escalation F1:",
    round(escalation_f1, 3)
)

print(
    "\nEscalation False Negatives:",
    len(false_negatives)
)

print(
    "\nSaved:",
    OUTPUT_FILE
)

print("\nIMPORTANT:")
print(
    "These metrics are based on the held-out 20% "
    "test set, not the training data."
)

print(
    "\nResponse-quality scores such as relevance, "
    "grounding, helpfulness, tone and hallucination "
    "still require manual evaluation."
)