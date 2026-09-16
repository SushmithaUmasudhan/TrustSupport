import pandas as pd

FILE = "response_evaluation.csv"

df = pd.read_csv(FILE)

false_negatives = df[
    (df["gold_escalate"] == "YES") &
    (df["predicted_escalate"] == "NO")
].copy()

print("=" * 70)
print("ESCALATION FAILURE ANALYSIS")
print("=" * 70)

print("\nTotal false negatives:", len(false_negatives))

for i, (_, row) in enumerate(false_negatives.iterrows(), start=1):
    print("\n" + "-" * 70)
    print(f"FALSE NEGATIVE {i}")
    print("-" * 70)

    print("Customer:")
    print(row["text"])

    print("\nGold intent:")
    print(row["gold_intent"])

    print("Predicted intent:")
    print(row["predicted_intent"])

    print("\nGold escalation:")
    print(row["gold_escalate"])

    print("Predicted escalation:")
    print(row["predicted_escalate"])

    print("\nRetrieved historical customer:")
    print(row["retrieved_customer"])

    print("\nRetrieved historical reply:")
    print(row["retrieved_reply"])

    print("\nRetrieval similarity:")
    print(row["retrieval_similarity"])

print("\n" + "=" * 70)
print("END")
print("=" * 70)