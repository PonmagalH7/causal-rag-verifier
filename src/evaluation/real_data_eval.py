"""
REAL-DATA EVALUATION: Run the full pipeline on real RAGTruth examples.

RAGTruth ships two files:
    - source_info.jsonl : the retrieved source documents/passages
    - response.jsonl    : LLM-generated responses + human-annotated
                           hallucination spans (ground truth)

This script:
    1. Loads a few real (source, response) pairs
    2. Extracts causal-looking sentences from the generated response (Track A)
    3. Runs each through retrieve -> verify -> correct (Track B)
    4. Prints the result next to RAGTruth's own human hallucination labels,
       so you can see - on REAL data - whether your system's verdicts line
       up with real human judgments.

FIRST RUN NOTE: RAGTruth's exact field names can vary slightly between
versions. This script prints the raw keys of the first example so you can
confirm the field names match, and adjust the FIELD NAMES section below
if needed - just tell Claude what it printed and it'll fix it in one edit.

Run: python src/evaluation/real_data_eval.py
"""

import json
import os
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss

# =======================================================================
# FIELD NAMES - adjust these if RAGTruth's schema differs from expected
# =======================================================================
SOURCE_ID_FIELD = "source_id"
SOURCE_TEXT_FIELD = "source_info"     # the passage/document text
RESPONSE_TEXT_FIELD = "response"       # the LLM-generated answer text
HALLUCINATION_FIELD = "labels"         # human-annotated hallucination spans

DATA_DIR = "data/raw/ragtruth/dataset"
NUM_EXAMPLES_TO_TEST = 40

# =======================================================================
# CAUSAL EXTRACTION (Deeksha's logic)
# =======================================================================
CAUSAL_MARKERS = [
    "because", "due to", "caused by", "led to", "as a result of", "resulted in"
]

def extract_causal_sentences(text):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    found = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(marker in sentence_lower for marker in CAUSAL_MARKERS):
            found.append(sentence.strip())
    return found


def classify_sentence(sentence, nli_classifier):
    temporal_markers = ["then", "after", "before", "later", "followed by", "subsequently"]
    causal_hypothesis = "This sentence states that one event caused another event."
    sentence_lower = sentence.lower()
    if any(marker in sentence_lower for marker in temporal_markers):
        return "temporal"
    result = nli_classifier(f"{sentence} </s></s> {causal_hypothesis}")[0]
    return "causal" if result["label"] == "ENTAILMENT" else "correlational"


# =======================================================================
# RETRIEVAL + VERIFICATION (Ponmagal's logic)
# =======================================================================
def retrieve_evidence(claim_text, documents, embed_model, top_k=5):
    if not documents:
        return []
    doc_embeddings = embed_model.encode(documents, convert_to_numpy=True)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    claim_embedding = embed_model.encode([claim_text], convert_to_numpy=True)
    k = min(top_k, len(documents))
    distances, indices = index.search(claim_embedding, k)
    return [documents[idx] for idx in indices[0]]


def build_evidence_graph(claim_text, retrieved_sentences, nli_model):
    """Individual sentence-level graph (kept for transparency/explanation display)."""
    label_map = {"ENTAILMENT": "supports", "CONTRADICTION": "contradicts", "NEUTRAL": "unrelated"}
    nodes = []
    for sentence in retrieved_sentences:
        result = nli_model(f"{sentence}</s></s>{claim_text}")[0]
        relation = label_map.get(result["label"], "unrelated")
        nodes.append({"sentence": sentence, "relation": relation, "confidence": round(result["score"], 4)})
    return nodes


def get_verdict_combined(claim_text, retrieved_sentences, nli_model):
    """
    IMPROVED verdict logic: instead of requiring ONE sentence to fully entail
    the claim (too strict - real claims often combine/paraphrase multiple
    sentences), this checks the claim against the COMBINED evidence block,
    and uses full probability scores rather than just the top label.

    This fixes the main failure mode found in the first evaluation run:
    faithful claims that paraphrase or combine several source sentences were
    being wrongly flagged NOT-ENOUGH-INFO because no single sentence matched
    word-for-word.
    """
    if not retrieved_sentences:
        return "NOT-ENOUGH-INFO", 0.0

    # Combine top evidence sentences into one block for richer context
    combined_evidence = " ".join(retrieved_sentences[:5])

    # Get full probability distribution (not just top label)
    result = nli_model(f"{combined_evidence}</s></s>{claim_text}", top_k=None)
    scores = {r["label"]: r["score"] for r in result}

    entailment_score = scores.get("ENTAILMENT", 0.0)
    contradiction_score = scores.get("CONTRADICTION", 0.0)

    # Thresholds tuned looser than default argmax - combined evidence with
    # paraphrased/combined claims rarely hits >0.9 entailment even when true,
    # so we trust a moderate entailment score if it clearly beats contradiction.
    if contradiction_score > 0.5 and contradiction_score > entailment_score:
        return "CONTRADICTED", contradiction_score
    elif entailment_score > 0.35 and entailment_score > contradiction_score:
        return "SUPPORTED", entailment_score
    else:
        return "NOT-ENOUGH-INFO", entailment_score


# =======================================================================
# LOAD REAL DATA
# =======================================================================
def find_claim_span(claim_text, response_text):
    """Find the character start/end position of a claim within the full response."""
    start = response_text.find(claim_text)
    if start == -1:
        return None, None
    return start, start + len(claim_text)


def check_human_hallucination_overlap(claim_start, claim_end, human_labels):
    """Return True if this claim's span overlaps ANY human-labeled hallucination span."""
    for label in human_labels:
        h_start = label.get("start", -1)
        h_end = label.get("end", -1)
        # Overlap if the spans intersect at all
        if claim_start < h_end and claim_end > h_start:
            return True, label.get("text", "")
    return False, None


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    source_path = os.path.join(DATA_DIR, "source_info.jsonl")
    response_path = os.path.join(DATA_DIR, "response.jsonl")

    if not os.path.exists(source_path) or not os.path.exists(response_path):
        print(f"ERROR: Could not find dataset files at {DATA_DIR}")
        print("Run this first:")
        print(f"  git clone https://github.com/ParticleMedia/RAGTruth.git {os.path.dirname(DATA_DIR)}")
        return

    print("Loading real RAGTruth data...")
    sources = load_jsonl(source_path)
    responses = load_jsonl(response_path)

    print(f"\nLoaded {len(sources)} source records and {len(responses)} response records.")
    print("\n--- First response record's raw keys (verify field names match) ---")
    print(list(responses[0].keys()))
    print("\n--- First source record's raw keys ---")
    print(list(sources[0].keys()))
    print("\nIf these don't match the FIELD NAMES section at the top of this script, tell Claude the exact keys shown above.\n")

    # Build a lookup: source_id -> source text
    source_lookup = {}
    for s in sources:
        sid = s.get(SOURCE_ID_FIELD)
        text = s.get(SOURCE_TEXT_FIELD, "")
        if sid is not None:
            source_lookup[sid] = text

    print("Loading models (cached after first run)...")
    nli_classifier = hf_pipeline("text-classification", model="roberta-large-mnli")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    tested = 0
    results = []
    for r in responses:
        if tested >= NUM_EXAMPLES_TO_TEST:
            break

        response_text = r.get(RESPONSE_TEXT_FIELD, "")
        source_id = r.get(SOURCE_ID_FIELD)
        human_labels = r.get(HALLUCINATION_FIELD, [])

        source_text = source_lookup.get(source_id, "")
        if not response_text or not source_text:
            continue

        causal_sentences = extract_causal_sentences(response_text)
        if not causal_sentences:
            continue

        # Split source into sentence-level "documents" for retrieval
        import re
        source_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', source_text) if s.strip()]

        print(f"\n{'='*70}")
        print(f"EXAMPLE {tested + 1} (source_id: {source_id})")
        print(f"{'='*70}")
        print(f"Human-annotated hallucinations in this response: {len(human_labels)}")

        for claim in causal_sentences:
            label = classify_sentence(claim, nli_classifier)
            if label != "causal":
                continue

            claim_start, claim_end = find_claim_span(claim, response_text)
            if claim_start is None:
                continue  # couldn't locate exact span, skip

            is_human_hallucination, human_span_text = check_human_hallucination_overlap(
                claim_start, claim_end, human_labels
            )

            retrieved = retrieve_evidence(claim, source_sentences, embed_model, top_k=8)
            verdict, confidence = get_verdict_combined(claim, retrieved, nli_classifier)

            # System says "unsupported" if verdict is anything but SUPPORTED
            system_says_hallucination = (verdict != "SUPPORTED")

            # Does system agree with human ground truth for THIS claim?
            agrees = (system_says_hallucination == is_human_hallucination)

            print(f"\n  CLAIM: {claim}")
            print(f"  SYSTEM VERDICT: {verdict} (confidence: {confidence:.3f})")
            print(f"  HUMAN GROUND TRUTH: {'HALLUCINATION' if is_human_hallucination else 'FAITHFUL'}")
            print(f"  AGREEMENT: {'YES' if agrees else 'NO'}")

            results.append({
                "source_id": source_id,
                "claim": claim,
                "system_verdict": verdict,
                "system_says_hallucination": system_says_hallucination,
                "human_ground_truth_hallucination": is_human_hallucination,
                "agrees_with_human": agrees,
            })

        tested += 1

    print(f"\n{'='*70}")
    print(f"SUMMARY - tested {tested} real examples, found {len(results)} causal claims to verify")
    print(f"{'='*70}")

    if results:
        agree_count = sum(1 for r in results if r["agrees_with_human"])
        accuracy = agree_count / len(results)

        true_positives = sum(1 for r in results if r["system_says_hallucination"] and r["human_ground_truth_hallucination"])
        false_positives = sum(1 for r in results if r["system_says_hallucination"] and not r["human_ground_truth_hallucination"])
        false_negatives = sum(1 for r in results if not r["system_says_hallucination"] and r["human_ground_truth_hallucination"])

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        print(f"\nAccuracy (agreement with human labels): {accuracy:.2%} ({agree_count}/{len(results)})")
        print(f"Precision (of flagged claims, how many were real hallucinations): {precision:.2%}")
        print(f"Recall (of real hallucinations, how many did we catch): {recall:.2%}")
        print(f"True Positives: {true_positives} | False Positives: {false_positives} | False Negatives: {false_negatives}")
    else:
        print("\nNo causal claims with matched spans found in this sample - increase NUM_EXAMPLES_TO_TEST and try again.")

    print(json.dumps(results, indent=2))

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/real_data_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to data/processed/real_data_eval_results.json")


if __name__ == "__main__":
    main()