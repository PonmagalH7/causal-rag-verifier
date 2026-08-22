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

## Repository Structure
