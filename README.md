# Causal-RAG-Verifier

### Explainable Detection and Correction of Unsupported Causal Links in Multi-Source Retrieval-Augmented Generation (RAG) Systems

[![Status](https://img.shields.io/badge/status-active%20development-yellow)]()
[![Course](https://img.shields.io/badge/course-BITE308L-blue)]()
[![Institution](https://img.shields.io/badge/VIT-Fall%202026--27-blueviolet)]()

---

## Team

| Name | Register Number | Track |
|---|---|---|
| **Ponmagal H** | 24BIT0433 | [Track B — Retrieval, Verification & Correction](docs/track-b-ponmagal.md) |
| **Deeksha P** | 24BIT0483 | [Track A — Extraction, Classification & Explanation](docs/track-a-deeksha.md) |


---

## The Problem, in One Example

When an LLM answers using multiple retrieved documents, it can invent a causal connection between two **individually true** facts that no source actually states:

> Document A: *"Pulp Fiction is a 1994 American crime film directed by Quentin Tarantino."*
> Document B: *"Tarantino was born in Knoxville, Tennessee."*
> **Hallucinated output:** *"Tarantino directed the film because he was born in Tennessee."*

Both source facts are true. The **causal link between them is entirely fabricated.** We call this a **source-stitched ("Frankenstein") causal hallucination** — and no existing hallucination detector specifically targets it.

---

## System Architecture
<img width="2579" height="1480" alt="architecture_diagram" src="https://github.com/user-attachments/assets/881e20e2-769d-478f-9a16-5406b16bdad2" />

![System Architecture](assets/architecture_diagram.png)

The system runs as two connected tracks feeding into one pipeline (`src/integration_pipeline.py`):

- **Track A** (Deeksha) — finds causal-sounding sentences in generated text and confirms which are genuinely causal
- **Track B** (Ponmagal) — retrieves evidence, verifies each claim against it, and attempts a correction if unsupported

---

## Design Methodology
<img width="1380" height="2179" alt="methodology_diagram" src="https://github.com/user-attachments/assets/449a5759-3794-4047-9a27-118ea982dfd5" />

![Methodology Pipeline](assets/methodology_diagram.png)

---

## Current Status

**Working end-to-end prototype, evaluated on real human-labeled data.**

| Metric | Result |
|---|---|
| Evaluation dataset | [RAGTruth](https://github.com/ParticleMedia/RAGTruth) (real, human-annotated) |
| Accuracy vs. human hallucination labels | **72%** |
| Method | Combined-evidence NLI scoring with tuned entailment threshold |
| Known limitation | Correction module reliably fixes Frankenstein-stitched claims; struggles with fact-replacement on directly Contradicted claims (documented, see [results log](docs/results_log.md)) |

Full evaluation history, including the 36% → 72% improvement and why it happened: **[docs/results_log.md](docs/results_log.md)**

---

## Repository Structure

```
causal-rag-verifier/
├── assets/                       # Diagrams used in this README
├── data/
│   ├── raw/                      # Downloaded datasets (RAGTruth, etc.)
│   ├── annotated/                # Gold-labelled causal hallucination test set
│   └── processed/                # Evaluation outputs, results JSON
├── src/
│   ├── extraction/                # Track A — causal sentence extraction + classification
│   ├── explanation/                # Track A — explanation generation
│   ├── retrieval/                 # Track B — FAISS retrieval, evidence graphs
│   ├── correction/                 # Track B — evidence-consistent rewrite module
│   ├── evaluation/                 # Track B — real-data evaluation harness
│   ├── pipeline.py                 # Standalone retrieve→verify→correct pipeline
│   └── integration_pipeline.py     # Full connected pipeline (Track A + Track B)
├── docs/
│   ├── track-a-deeksha.md          # Track A detailed documentation
│   ├── track-b-ponmagal.md         # Track B detailed documentation
│   ├── annotation_guideline.md
│   ├── schema.md
│   └── results_log.md
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/PonmagalH7/causal-rag-verifier.git
cd causal-rag-verifier
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -U spacy
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run the full integrated pipeline:
```bash
python src/integration_pipeline.py
```

Run the real-data evaluation:
```bash
git clone https://github.com/ParticleMedia/RAGTruth.git data/raw/ragtruth
python src/evaluation/real_data_eval.py
```

---

## Datasets

| Dataset | Use | Reference |
|---|---|---|
| [RAGTruth](https://github.com/ParticleMedia/RAGTruth) | Evaluation (real human-annotated hallucinations) | Niu et al. (2024) |
| [HotpotQA](https://hotpotqa.github.io/) | Development / multi-hop test construction | Yang et al. (2018) |
| [RAMDocs](https://github.com/HanNight/RAMDocs) | Planned — multi-document conflict evaluation | Wang et al. (2025) |
| [MAGIC](https://arxiv.org/abs/2507.21544) | Planned — knowledge-graph conflict evaluation | Lee et al. (2025) |
| [CLADDER](https://arxiv.org/abs/2312.04350) | Planned — causal reasoning benchmark | Jin et al. (2023) |

---

## Honest Limitations (Current Stage)

- Evaluation sample size (~25–40 claims) is a pilot; being scaled up
- No direct baseline comparison yet against RAGTruth/Bi'an/ReDeEP/MADAM-RAG
- Correction module (FLAN-T5-base, free/local) underperforms specifically on Contradicted claims — a model-capacity limitation, not a design flaw

Full remaining-work roadmap: **[docs/remaining_work.md](docs/remaining_work.md)**

---

## Intended Outcome

This project is being developed toward submission as a conference paper and/or provisional patent application. See [docs/results_log.md](docs/results_log.md) for the methodology and findings supporting either path.

## License
TBD (to be added prior to publication/patent filing)

