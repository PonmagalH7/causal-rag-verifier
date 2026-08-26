# Causal-RAG-Verifier

### Explainable Detection and Correction of Unsupported Causal Links in Multi-Source Retrieval-Augmented Generation (RAG) Systems

## Team
| Name | Register Number | Role |
|---|---|---|
| Ponmagal H | 24BIT0433 | Pipeline, Retrieval, Evidence Graphs, Correction Module, Evaluation |
| Deeksha P | 24BIT0483 | Causal Extraction, NLI Verification, Explanation Generation, Annotation |

**Course:** BITE308L — Artificial Intelligence
**Institution:** School of Computer Science Engineering and Information Systems, VIT
**Semester:** Fall 2026-27

## Overview
Retrieval-Augmented Generation (RAG) grounds LLM outputs in retrieved documents to reduce hallucination. However, when multiple sources are combined, models often synthesize causal claims ("X causes Y") that no single source actually supports — a failure mode we call **source-stitched ("Frankenstein") causal hallucination**.

This project builds a pipeline that:
1. **Extracts** causal assertions from LLM-generated answers (dependency + discourse-marker parsing, LLM-based causal classifier)
2. **Verifies** each claim against multi-source evidence graphs using NLI and evidence-aggregation scoring
3. **Explains** why a claim is unsupported (missing, contradictory, or merely correlational evidence)
4. **Corrects** the claim with an evidence-consistent rewrite

# Results Log

This file tracks evaluation runs, methodology changes, and findings.
Written to double as the backbone of the paper's Methodology/Results/
Limitations sections later.

---

## Run 1 — Initial real-data evaluation (baseline)

**Method:** For each retrieved evidence sentence, individually check whether
it entails the claim via NLI (`roberta-large-mnli`). Verdict = SUPPORTED
only if at least one sentence's top NLI label is ENTAILMENT.

**Sample:** 40 real (source, response) pairs from RAGTruth, yielding 25
extracted causal claims with human ground-truth hallucination labels.

**Results:**
- Accuracy: 36.00% (9/25)
- Precision: 5.88%
- Recall: 100.00%
- True Positives: 1, False Positives: 16, False Negatives: 0

**Finding:** The system correctly caught every real hallucination in the
sample (100% recall), but severely over-flagged faithful claims as
unsupported (5.88% precision). Root cause: single-sentence NLI entailment
is too strict for claims that paraphrase or combine information spread
across multiple source sentences — a faithful but reworded claim rarely
achieves ENTAILMENT status against any single source sentence.

---

## Run 2 — Combined-evidence, threshold-tuned verdict

**Method changed to:** Concatenate the top 5 retrieved evidence sentences
into a single evidence block. Run NLI once against the combined block
(claim as hypothesis), using the full probability distribution (not just
the top label). Verdict thresholds:
- CONTRADICTED if contradiction score > 0.5 and exceeds entailment score
- SUPPORTED if entailment score > 0.35 and exceeds contradiction score
- Otherwise NOT-ENOUGH-INFO

**Sample:** Same 40 pairs / 25 causal claims as Run 1.

**Results:**
- Accuracy: 72.00% (18/25)
- Precision: 0.00%
- Recall: 0.00%
- True Positives: 0, False Positives: 6, False Negatives: 1

**Finding:** Accuracy roughly doubled by giving the NLI model full
multi-sentence context and a more realistic entailment threshold. However,
recall dropped to 0% on this sample: the single real hallucination present
(a plausible-sounding but factually altered date/detail claim) was now
scored as SUPPORTED because its wording closely matched the combined
evidence despite an underlying factual alteration.

**Important caveat:** With only 1 true hallucination in this 25-claim
sample, precision/recall are not statistically stable at this sample size
- a single miss swings recall from 100% to 0%. Accuracy (72%) is the more
meaningful number here, but **a larger evaluation sample is needed before
citing precision/recall figures in any paper or patent submission.**

---

## Known Limitations (as of Run 2)

1. **Correction module** (FLAN-T5-base): reliably rewrites Frankenstein-
   stitched claims (drops unsupported causal language) but fails to
   perform fact-replacement on directly Contradicted claims (e.g. wrong
   nationality claims) - repeatedly echoes the original, uncorrected claim
   even with an explicit worked example in the prompt. This appears to be
   a model-capacity limitation rather than a prompt-design issue, since
   two different prompt strategies produced identical failures.

2. **Sample size**: current evaluation (25 claims) is a pilot, not a
   publication-grade evaluation. Needs to scale to 150-200+ claims for
   statistically meaningful precision/recall.

3. **No baseline comparison yet**: related work (RAGTruth, Bi'an, ReDeEP,
   MADAM-RAG) has not been run head-to-head against this system on the
   same data.

4. **Recall/precision tradeoff is threshold-dependent**: Run 1 favored
   recall, Run 2 favored precision/accuracy. The right operating point
   depends on the target use case (e.g., high-stakes domains like
   healthcare may prefer higher recall even at lower precision).

5. **Single dataset**: only evaluated on RAGTruth so far. Abstract's
   dataset list includes RAMDocs, MAGIC, CLADDER, HotpotQA - not yet used
   for evaluation.

---

## Next Steps
- [ ] Re-run evaluation on 150-200+ claims for statistical stability
- [ ] Explore a middle-ground threshold that improves recall without
      losing Run 2's accuracy gains
- [ ] Wire in Deeksha's explanation generator to the evaluation output
- [ ] Attempt correction fix with a larger model (budget permitting) or
      document the limitation as final for this iteration
- [ ] Run baseline comparison against at least one related-work system
