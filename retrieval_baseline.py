import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_FILE = "amazon_support.csv"

print("Loading Amazon support data...")

df = pd.read_csv(INPUT_FILE)

# AmazonHelp replies
amazon = df[
    df["author_id"].astype(str).str.strip().str.lower() == "amazonhelp"
].copy()

# Customer messages
customers = df[
    df["inbound"].astype(str).str.strip().str.lower() == "true"
].copy()

customers = customers[
    customers["text"].notna()
].copy()

customers["text"] = customers["text"].astype(str).str.strip()

print("Customer messages:", len(customers))
print("AmazonHelp replies:", len(amazon))

# Create mapping:
# AmazonHelp tweet ID -> reply text
reply_map = dict(
    zip(
        amazon["tweet_id"].astype(str),
        amazon["text"].astype(str)
    )
)

# Keep only customer messages that have an AmazonHelp reply
pairs = []

for _, row in customers.iterrows():

    response_ids = str(row["response_tweet_id"]).split(",")

    reply = None

    for response_id in response_ids:
        response_id = response_id.strip()

        if response_id in reply_map:
            reply = reply_map[response_id]
            break

    if reply:
        pairs.append({
            "customer_text": row["text"],
            "amazon_reply": reply
        })

pairs = pd.DataFrame(pairs)

print("Customer → AmazonHelp pairs:", len(pairs))

# Remove duplicate customer messages
pairs = pairs.drop_duplicates(subset=["customer_text"])

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

customer_vectors = vectorizer.fit_transform(
    pairs["customer_text"]
)

print("TF-IDF matrix:", customer_vectors.shape)


def retrieve_reply(query, top_k=3):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        customer_vectors
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "similarity": round(float(similarities[index]), 3),
            "customer_message": pairs.iloc[index]["customer_text"],
            "amazon_reply": pairs.iloc[index]["amazon_reply"]
        })

    return results


# Test example
query = "My package has not arrived yet. Can you help?"

print("\n" + "=" * 60)
print("RETRIEVAL TEST")
print("=" * 60)

print("\nCustomer:")
print(query)

results = retrieve_reply(query)

for i, result in enumerate(results, 1):

    print("\n--- Result", i, "---")
    print("Similarity:", result["similarity"])
    print("Historical customer:")
    print(result["customer_message"])
    print("\nHistorical AmazonHelp reply:")
    print(result["amazon_reply"])