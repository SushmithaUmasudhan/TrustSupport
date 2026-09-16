import pandas as pd

INPUT_FILE = "amazon_support.csv"
OUTPUT_FILE = "amazon_intent_dataset.csv"

df = pd.read_csv(INPUT_FILE)

# Keep customer messages only
customer = df[
    df["inbound"].astype(str).str.strip().str.lower() == "true"
].copy()

customer = customer[
    customer["text"].notna()
].copy()

customer["text"] = customer["text"].astype(str).str.strip()

# Remove extremely short messages
customer = customer[customer["text"].str.len() >= 15]

# Clean obvious duplicates
customer = customer.drop_duplicates(subset=["text"])

print("Customer messages available:", len(customer))

# --------------------------------------------------
# Initial keyword-based intent assignment
# --------------------------------------------------

def assign_intent(text):
    t = text.lower()

    if any(x in t for x in [
        "delivery", "delivered", "deliver", "shipping",
        "shipment", "package", "parcel", "tracking",
        "late", "delayed", "arrive"
    ]):
        return "Order & Delivery"

    if any(x in t for x in [
        "refund", "return", "returned", "returning",
        "money back", "refunds"
    ]):
        return "Returns & Refunds"

    if any(x in t for x in [
        "payment", "paid", "pay", "credit card",
        "debit card", "amazon pay", "charged",
        "charge", "transaction"
    ]):
        return "Payment & Amazon Pay"

    if any(x in t for x in [
        "account", "login", "locked", "password",
        "security", "verification", "suspended"
    ]):
        return "Account & Security"

    if any(x in t for x in [
        "fake", "counterfeit", "wrong item", "damaged",
        "broken product", "defective", "product quality"
    ]):
        return "Product Problems"

    if any(x in t for x in [
        "prime membership", "amazon prime", "prime",
        "prime music", "prime benefit"
    ]):
        return "Prime Membership & Benefits"

    if any(x in t for x in [
        "app", "mobile", "kindle", "firestick",
        "fire stick", "device", "exchange"
    ]):
        return "App / Device / Technical"

    if any(x in t for x in [
        "prime video", "video", "subtitle", "subtitles",
        "streaming", "episode", "movie"
    ]):
        return "Prime Video / Streaming"

    return "Other"


customer["intent"] = customer["text"].apply(assign_intent)

# Keep only the 8 intended categories
customer = customer[
    customer["intent"] != "Other"
].copy()

# --------------------------------------------------
# Escalation signal
# --------------------------------------------------

def escalation_reason(text):
    t = text.lower()

    if any(x in t for x in [
        "multiple times", "again and again",
        "still no", "still not", "no response",
        "nobody helped", "not resolved",
        "already contacted", "already called",
        "tried everything"
    ]):
        return "Repeated or unresolved support attempt"

    if any(x in t for x in [
        "fraud", "unauthorized", "stolen",
        "security concern", "someone used"
    ]):
        return "Potential security or unauthorized activity"

    if any(x in t for x in [
        "fake", "counterfeit"
    ]):
        return "Potential counterfeit product"

    return ""


customer["escalate"] = customer["text"].apply(
    lambda x: "YES" if escalation_reason(x) else "NO"
)

customer["escalation_reason"] = customer["text"].apply(
    escalation_reason
)

# Save
customer.to_csv(OUTPUT_FILE, index=False)

print()
print("=" * 60)
print("INTENT DATASET CREATED")
print("=" * 60)

print("\nIntent counts:")
print(customer["intent"].value_counts())

print("\nEscalation counts:")
print(customer["escalate"].value_counts())

print("\nCreated:", OUTPUT_FILE)