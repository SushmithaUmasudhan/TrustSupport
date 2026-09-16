import pandas as pd

INPUT_FILE = "twcs.csv"
OUTPUT_FILE = "amazon_support.csv"

print("Reading original dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total rows:", len(df))

# AmazonHelp tweets
amazon = df[
    df["author_id"].astype(str).str.strip().str.lower() == "amazonhelp"
].copy()

print("AmazonHelp tweets:", len(amazon))

# IDs of AmazonHelp replies
amazon_ids = set(
    amazon["tweet_id"].astype(str)
)

# Find customer tweets whose response_tweet_id
# points to an AmazonHelp reply
def contains_amazon_reply(value):
    if pd.isna(value):
        return False

    ids = str(value).split(",")

    return any(tweet_id.strip() in amazon_ids for tweet_id in ids)


customers = df[
    df["response_tweet_id"].apply(contains_amazon_reply)
].copy()

print("Customer tweets:", len(customers))

# Combine customer messages + AmazonHelp replies
result = pd.concat(
    [customers, amazon],
    ignore_index=True
).drop_duplicates(subset="tweet_id")

print("Final conversation rows:", len(result))

result.to_csv(OUTPUT_FILE, index=False)

print("Created:", OUTPUT_FILE)