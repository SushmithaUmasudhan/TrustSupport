# TrustSupport: An Evidence-Grounded AI Customer Support Agent

TrustSupport is an AI customer-support prototype built using the Customer Support on Twitter (TWCS) dataset. The system focuses on Amazon customer-support conversations and combines intent classification, historical response retrieval, response generation, and human escalation.

## Project Overview

The system follows this pipeline:

Customer Message → Intent Classification → Historical Retrieval → Response Generation → Escalation Decision → Final Response

The project defines 8 customer-support intents:

1. Order & Delivery
2. Returns & Refunds
3. Payment & Amazon Pay
4. Account & Security
5. Product Problems
6. Prime Membership & Benefits
7. App / Device / Technical
8. Prime Video / Streaming

## Dataset

The project uses the Customer Support on Twitter (TWCS) dataset.

The original dataset contains approximately 2.8 million tweets. Amazon-related conversations were extracted using AmazonHelp tweets and customer tweets linked through response_tweet_id.

Due to repository size limitations, the original dataset is not included in this repository.

## Gold Dataset

A manually labelled gold dataset of 200 customer-support examples was created.

The gold dataset was used to evaluate intent classification and escalation performance.

## Methods

### Intent Classification

Two TF-IDF + Logistic Regression baselines were evaluated:

- Unweighted Logistic Regression
- Balanced Logistic Regression using class weights

The balanced model was used in the final prototype.

### Historical Response Retrieval

TF-IDF similarity is used to retrieve relevant historical customer-support responses from the TWCS dataset.

Exact self-matches are excluded during evaluation to reduce retrieval leakage.

### Response Generation

For non-escalation cases, the system returns a relevant historical AmazonHelp response as an evidence-grounded response.

For escalation cases, the system recommends human support.

### Human Escalation

The escalation classifier identifies cases that may require human intervention, particularly unresolved, repeated, or potentially sensitive customer-support issues.

## Results

### Intent Classification

The final balanced intent classifier achieved:

- Accuracy: 47.5%
- Macro-F1: 0.419

Evaluation was performed on a 40-example held-out test set.

### Escalation Classification

The escalation classifier achieved:

- Accuracy: 60.0%
- Precision: 57.1%
- Recall: 23.5%
- F1: 0.333
- False Negatives: 13

The low escalation recall shows that accuracy alone is not sufficient for evaluating safety-oriented escalation.

## Failure Analysis

The main observed failure modes were:

1. Repeated or unresolved support problems were sometimes not escalated.
2. Multi-issue customer messages confused the classifier.
3. Frustration was sometimes expressed indirectly.
4. Intent misclassification contributed to some escalation failures.
5. The small labelled dataset limited the diversity of escalation examples.

## Repository Files

- `extract_amazon.py` — extracts AmazonHelp conversations from TWCS.
- `build_intents.py` — creates initial intent and escalation labels.
- `retrieval_baseline.py` — implements historical response retrieval.
- `complete_agent.py` — runs the complete support-agent pipeline.
- `evaluate_responses.py` — evaluates intent and escalation performance.
- `analyze_escalation.py` — analyzes escalation false negatives.
- `confusionmatrix.png` — intent-classification confusion matrix.

## Limitations

The manually labelled dataset contains only 200 examples, with a relatively small held-out test set of 40 examples. Therefore, the reported metrics should be interpreted as prototype-level results rather than broad estimates of real-world performance.

## Future Work

Future improvements could include:

- Increasing the size and diversity of the labelled dataset.
- Using transformer-based intent classification.
- Improving semantic retrieval.
- Adding explicit human evaluation of response relevance, grounding, helpfulness, and tone.
- Developing a higher-recall escalation mechanism.


