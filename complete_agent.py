import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD GOLD DATA
# ============================================================

print("Loading gold dataset...")

gold = pd.read_csv("amazon_gold_200_labelled.csv")

X = gold["text"].fillna("").astype(str)
y_intent = gold["gold_intent"].fillna("").astype(str)
y_escalate = gold["gold_escalate"].fillna("").astype(str)


# ============================================================
# 2. TRAIN INTENT CLASSIFIER
# ============================================================

print("Training intent classifier...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_intent,
    test_size=0.2,
    random_state=42,
    stratify=y_intent
)

intent_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_intent = intent_vectorizer.fit_transform(X_train)

intent_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

intent_model.fit(
    X_train_intent,
    y_train
)


# ============================================================
# 3. TRAIN ESCALATION CLASSIFIER
# ============================================================

print("Training escalation classifier...")

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X,
    y_escalate,
    test_size=0.2,
    random_state=42,
    stratify=y_escalate
)

escalate_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_escalate = escalate_vectorizer.fit_transform(X_train_e)

escalate_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

escalate_model.fit(
    X_train_escalate,
    y_train_e
)


# ============================================================
# 4. LOAD HISTORICAL AMAZON SUPPORT DATA
# ============================================================

print("Loading historical Amazon support data...")

data = pd.read_csv("amazon_support.csv")

amazon = data[
    data["author_id"].astype(str).str.strip().str.lower()
    == "amazonhelp"
].copy()

customers = data[
    data["inbound"].astype(str).str.strip().str.lower()
    == "true"
].copy()

customers = customers[
    customers["text"].notna()
].copy()

customers["text"] = (
    customers["text"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 5. BUILD CUSTOMER → AMAZONHELP REPLY PAIRS
# ============================================================

print("Building historical conversation pairs...")

reply_map = dict(
    zip(
        amazon["tweet_id"].astype(str),
        amazon["text"].astype(str)
    )
)

pairs = []

for _, row in customers.iterrows():

    response_ids = str(
        row["response_tweet_id"]
    ).split(",")

    for response_id in response_ids:

        response_id = response_id.strip()

        if response_id in reply_map:

            pairs.append({
                "customer_text": row["text"],
                "amazon_reply": reply_map[response_id]
            })

            break


pairs = pd.DataFrame(pairs)

pairs = pairs.drop_duplicates(
    subset=["customer_text"]
).reset_index(drop=True)

print(
    "Historical customer/reply pairs:",
    len(pairs)
)


# ============================================================
# 6. CLASSIFY HISTORICAL CUSTOMER MESSAGES
# ============================================================

print("Classifying historical conversations by intent...")

pairs["historical_intent"] = intent_model.predict(
    intent_vectorizer.transform(
        pairs["customer_text"].astype(str)
    )
)

print("Historical intent index created.")


# ============================================================
# 7. CREATE TF-IDF RETRIEVAL INDEX
# ============================================================

print("Creating retrieval index...")

retrieval_vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

historical_vectors = retrieval_vectorizer.fit_transform(
    pairs["customer_text"]
)


# ============================================================
# 8. IMPORTANT SUPPORT PHRASES
# ============================================================

# These phrases help the retriever distinguish issues that
# may share general words such as "Amazon", "account", etc.

SUPPORT_PHRASES = {
    "unrecognized_charge": [
        "don't recognize this charge",
        "do not recognize this charge",
        "unrecognized charge",
        "unknown charge",
        "unknown transaction",
        "charge i don't recognize",
        "charge I do not recognize",
        "charge on my bank account",
        "charge on my account",
        "unauthorized charge",
        "unexpected charge"
    ],

    "hacked_account": [
        "account hacked",
        "amazon account hacked",
        "someone hacked my account",
        "account compromised",
        "someone accessed my account"
    ],

    "delivery_problem": [
        "package has not arrived",
        "package hasn't arrived",
        "order has not arrived",
        "order hasn't arrived",
        "package not delivered",
        "delivery delayed",
        "late delivery",
        "where is my package"
    ],

    "refund_problem": [
        "refund not received",
        "where is my refund",
        "waiting for refund",
        "refund missing",
        "need a refund"
    ],

    "damaged_product": [
        "product arrived damaged",
        "item arrived damaged",
        "received damaged",
        "broken item",
        "damaged item",
        "defective product"
    ],

    "prime_video": [
        "prime video",
        "cannot stream",
        "can't stream",
        "video won't play",
        "video not playing",
        "streaming problem",
        "streaming issue"
    ],

    "prime_membership": [
        "prime membership",
        "amazon prime",
        "prime subscription",
        "prime renewal",
        "prime charge"
    ]
}


# ============================================================
# 9. NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 10. FIND IMPORTANT PHRASES
# ============================================================

def find_phrase_matches(query, historical_text):

    query_normalized = normalize_text(query)

    historical_normalized = normalize_text(
        historical_text
    )

    matched_categories = []

    for category, phrases in SUPPORT_PHRASES.items():

        for phrase in phrases:

            phrase_normalized = normalize_text(
                phrase
            )

            if phrase_normalized in query_normalized:

                if phrase_normalized in historical_normalized:

                    matched_categories.append(
                        category
                    )

    return matched_categories


# ============================================================
# 11. PREDICT INTENT
# ============================================================

def predict_intent(query):

    vector = intent_vectorizer.transform(
        [query]
    )

    return intent_model.predict(vector)[0]


# ============================================================
# 12. PREDICT ESCALATION
# ============================================================

def predict_escalation(query):

    vector = escalate_vectorizer.transform(
        [query]
    )

    return escalate_model.predict(vector)[0]


# ============================================================
# 13. HYBRID RETRIEVAL
# ============================================================

def retrieve_evidence(
    query,
    predicted_intent,
    top_k=3
):

    # --------------------------------------------------------
    # STEP 1
    # Only retrieve examples from the predicted intent.
    # --------------------------------------------------------

    matching_indices = pairs.index[
        pairs["historical_intent"] == predicted_intent
    ].tolist()

    if not matching_indices:

        return []


    # --------------------------------------------------------
    # STEP 2
    # Calculate normal TF-IDF similarity.
    # --------------------------------------------------------

    query_vector = retrieval_vectorizer.transform(
        [query]
    )

    matching_vectors = historical_vectors[
        matching_indices
    ]

    similarities = cosine_similarity(
        query_vector,
        matching_vectors
    )[0]


    # --------------------------------------------------------
    # STEP 3
    # Calculate a hybrid score.
    #
    # Base score = TF-IDF similarity
    #
    # Phrase match = additional evidence that the historical
    # example is talking about the same specific problem.
    # --------------------------------------------------------

    scored_results = []

    for position, original_index in enumerate(
        matching_indices
    ):

        customer_text = pairs.loc[
            original_index,
            "customer_text"
        ]

        similarity = float(
            similarities[position]
        )

        phrase_matches = find_phrase_matches(
            query,
            customer_text
        )

        # Small bonus for matching important phrases
        phrase_bonus = min(
            len(phrase_matches) * 0.15,
            0.30
        )

        hybrid_score = (
            similarity +
            phrase_bonus
        )

        scored_results.append({
            "index": original_index,
            "similarity": similarity,
            "phrase_bonus": phrase_bonus,
            "hybrid_score": hybrid_score,
            "phrase_matches": phrase_matches
        })


    # --------------------------------------------------------
    # STEP 4
    # Sort using the hybrid score.
    # --------------------------------------------------------

    scored_results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )


    # --------------------------------------------------------
    # STEP 5
    # Return top results.
    # --------------------------------------------------------

    results = []

    for item in scored_results[:top_k]:

        index = item["index"]

        results.append({
            "similarity": round(
                item["similarity"],
                3
            ),

            "hybrid_score": round(
                item["hybrid_score"],
                3
            ),

            "phrase_matches": item[
                "phrase_matches"
            ],

            "customer": pairs.loc[
                index,
                "customer_text"
            ],

            "reply": pairs.loc[
                index,
                "amazon_reply"
            ]
        })

    return results


# ============================================================
# 14. CLEAN HISTORICAL REPLY
# ============================================================

def clean_reply(reply):

    reply = str(reply)

    # Remove Twitter agent signature
    reply = re.sub(
        r"\s*\^[A-Za-z]{1,4}\s*$",
        "",
        reply
    )

    # Remove username at beginning
    reply = re.sub(
        r"^@\w+\s*",
        "",
        reply
    )

    # Remove excessive whitespace
    reply = re.sub(
        r"\s+",
        " ",
        reply
    )

    return reply.strip()


# ============================================================
# 15. GENERATE RESPONSE
# ============================================================

def generate_response(
    intent,
    escalation,
    evidence
):

    # --------------------------------------------------------
    # Escalation case
    # --------------------------------------------------------

    if escalation == "YES":

        return (
            "This issue may require a human support agent "
            "to investigate your specific account or order. "
            "Please contact Amazon Customer Support directly."
        )


    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if not evidence:

        return (
            "We couldn't find a closely matching historical "
            "support example. Please contact Amazon Customer "
            "Support for assistance."
        )


    # --------------------------------------------------------
    # Evidence exists
    # --------------------------------------------------------

    best_reply = clean_reply(
        evidence[0]["reply"]
    )

    return best_reply


# ============================================================
# 16. RUN COMPLETE AGENT
# ============================================================

def run_agent(query):

    print(
        "\nProcessing customer message..."
    )


    # --------------------------------------------------------
    # STEP 1: INTENT
    # --------------------------------------------------------

    intent = predict_intent(
        query
    )


    # --------------------------------------------------------
    # STEP 2: ESCALATION
    # --------------------------------------------------------

    escalation = predict_escalation(
        query
    )


    # --------------------------------------------------------
    # STEP 3: HYBRID RETRIEVAL
    # --------------------------------------------------------

    evidence = retrieve_evidence(
        query,
        intent,
        top_k=3
    )


    # --------------------------------------------------------
    # STEP 4: RESPONSE
    # --------------------------------------------------------

    response = generate_response(
        intent,
        escalation,
        evidence
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("AMAZON CUSTOMER SUPPORT AGENT")
    print("=" * 70)


    print("\nCustomer:")
    print(query)


    print("\nPredicted Intent:")
    print(intent)


    print("\nEscalation Decision:")

    if escalation == "YES":

        print(
            "HUMAN ESCALATION"
        )

    else:

        print(
            "AUTO-HANDLE"
        )


    print("\nHistorical Evidence:")


    if not evidence:

        print(
            "No historical evidence found."
        )

    else:

        for i, item in enumerate(
            evidence,
            start=1
        ):

            print(
                f"\n--- Evidence {i} ---"
            )

            print(
                "TF-IDF similarity:",
                item["similarity"]
            )

            print(
                "Hybrid score:",
                item["hybrid_score"]
            )

            if item["phrase_matches"]:

                print(
                    "Specific phrase matches:",
                    ", ".join(
                        item["phrase_matches"]
                    )
                )

            print(
                "\nHistorical customer:"
            )

            print(
                item["customer"]
            )

            print(
                "\nAmazonHelp reply:"
            )

            print(
                item["reply"]
            )


    print(
        "\n" + "-" * 70
    )


    print(
        "SUGGESTED CUSTOMER RESPONSE:"
    )

    print(
        response
    )


    print(
        "=" * 70
    )


# ============================================================
# 17. START PROGRAM
# ============================================================

if __name__ == "__main__":

    print(
        "\nAmazon Support Agent is ready!"
    )

    query = input(
        "\nEnter a customer message: "
    )

    if query.strip():

        run_agent(
            query.strip()
        )

    else:

        print(
            "Please enter a customer message."
        )