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
A majority-class baseline was also evaluated by always predicting the most common intent, Order & Delivery.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority-class baseline | 30.0% | ≈0.058 |
| TF-IDF + Logistic Regression | 30.0% | 0.060 |
| Balanced TF-IDF + Logistic Regression | 50.0% | 0.420 |
| Final held-out evaluation | 47.5% | 0.419 |

The unweighted model performed almost the same as the trivial majority baseline, while class balancing substantially improved macro-F1.

Evaluation was performed on a 40-example held-out test set.

### Escalation Classification

The escalation classifier achieved:

- Accuracy: 60.0%
- Precision: 57.1%
- Recall: 23.5%
- F1: 0.333
- False Negatives: 13

The low escalation recall shows that accuracy alone is not sufficient for evaluating safety-oriented escalation.

### LLM-as-Judge Evaluation

A structured LLM-as-judge rubric evaluated Relevance, Grounding, Helpfulness, Tone, Hallucination, and Overall quality using 1–5 scores.

Twenty examples were independently rated by a human using the same rubric. The LLM judge achieved 80% exact agreement with human overall ratings, with a quadratic Cohen's kappa of 0.916.

Individual dimension agreement was lower, so the LLM judge was used as a supplementary evaluator.

## Failure Analysis

The main observed failure modes were:

1. Repeated or unresolved support problems were sometimes not escalated.
2. Multi-issue customer messages confused the classifier.
3. Frustration was sometimes expressed indirectly.
4. Intent misclassification contributed to some escalation failures.
5. The small labelled dataset limited the diversity of escalation examples.

## Decision Log

1. Selected AmazonHelp for its large support-interaction volume.
2. Used `response_tweet_id` to reconstruct customer-brand relationships.
3. Defined eight intents to keep classification focused.
4. Treated unresolved support as an escalation signal.
5. Created 200 hand-labelled examples for evaluation.
6. Used stratified splitting to handle class imbalance.
7. Compared unweighted and balanced Logistic Regression.
8. Selected the balanced classifier based on improved macro-F1.
9. Used TF-IDF for historical-response retrieval.
10. Excluded self-matches to prevent retrieval leakage.
11. Grounded responses in historical AmazonHelp replies.
12. Analysed escalation false negatives as safety failures.
13. Validated the LLM judge against human ratings.
14. Used held-out evaluation for final results.
15. Reported limitations due to the small labelled dataset.

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
- Developing a higher-recall escalation mechanism.
- Expanding human and LLM response-quality evaluation to a larger sample.

## Reproduction

1. Download the TWCS dataset and place `twcs.csv` in the project folder.
2. Run:

```bash
python extract_amazon.py
python build_intents.py
python complete_agent.py
python evaluate_responses.py
The evaluation reports held-out intent and escalation metrics. The original TWCS dataset is not included in the repository because of its size.


