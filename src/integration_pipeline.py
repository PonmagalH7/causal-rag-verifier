"""
INTEGRATION: Track A (Deeksha) + Track B (Ponmagal) connected end to end.

This script simulates the full real-world scenario:
    1. An LLM generates an ANSWER paragraph (may contain hallucinated causal claims)
    2. EXTRACT   - pull out sentences that look like causal claims (Deeksha's logic)
    3. CLASSIFY  - confirm which ones are truly causal vs correlational/temporal (Deeksha's logic)
    4. RETRIEVE  - for each confirmed causal claim, find relevant evidence (Ponmagal's logic)
    5. VERIFY    - check if the evidence actually supports the claim (Ponmagal's logic)
    6. CORRECT   - if unsupported, generate a corrected rewrite (Ponmagal's logic)

Run: python src/integration_pipeline.py
"""

import json
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss


# =======================================================================
# TRACK A: EXTRACTION (based on Deeksha's extract_claims.py logic)
# =======================================================================
CAUSAL_MARKERS = [
    "because", "due to", "caused by", "led to", "as a result of", "resulted in"
]

def extract_causal_sentences(sentences):
    """Return only sentences that contain a causal discourse marker."""
    found = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(marker in sentence_lower for marker in CAUSAL_MARKERS):
            found.append(sentence)
    return found


# =======================================================================
# TRACK A: CLASSIFICATION (based on Deeksha's classify_causal.py logic)
# =======================================================================
TEMPORAL_MARKERS = [
    "then", "after", "before", "later", "followed by", "subsequently"
]
CAUSAL_HYPOTHESIS = "This sentence states that one event caused another event."

def classify_sentence(sentence, nli_classifier):
    """Label a sentence as causal, correlational, or temporal."""
    sentence_lower = sentence.lower()
    if any(marker in sentence_lower for marker in TEMPORAL_MARKERS):
        return "temporal"

    result = nli_classifier(f"{sentence} </s></s> {CAUSAL_HYPOTHESIS}")[0]
    if result["label"] == "ENTAILMENT":
        return "causal"
    return "correlational"


# =======================================================================
# TRACK B: RETRIEVE (Ponmagal's build_index.py logic)
# =======================================================================
def retrieve_evidence(claim_text, documents, source_titles, embed_model, top_k=5):
    doc_embeddings = embed_model.encode(documents, convert_to_numpy=True)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)

    claim_embedding = embed_model.encode([claim_text], convert_to_numpy=True)
    distances, indices = index.search(claim_embedding, min(top_k, len(documents)))

    retrieved = []
    for idx in indices[0]:
        retrieved.append({"sentence": documents[idx], "source": source_titles[idx]})
    return retrieved


# =======================================================================
# TRACK B: VERIFY (Ponmagal's evidence_graph.py logic)
# =======================================================================
def build_evidence_graph(claim_text, retrieved_sentences, nli_model):
    label_map = {
        "ENTAILMENT": "supports",
        "CONTRADICTION": "contradicts",
        "NEUTRAL": "unrelated",
    }
    evidence_nodes = []
    for rank, item in enumerate(retrieved_sentences):
        sentence = item["sentence"]
        source = item["source"]
        result = nli_model(f"{sentence}</s></s>{claim_text}")[0]
        relation = label_map.get(result["label"], "unrelated")
        evidence_nodes.append({
            "node_id": f"node_{rank+1}",
            "source_doc": source,
            "sentence_text": sentence,
            "relation_to_claim": relation,
            "nli_confidence": round(result["score"], 4),
        })
    return evidence_nodes


def get_verdict(evidence_nodes):
    supports = [n for n in evidence_nodes if n["relation_to_claim"] == "supports"]
    contradicts = [n for n in evidence_nodes if n["relation_to_claim"] == "contradicts"]
    if contradicts:
        return "CONTRADICTED"
    elif not supports:
        return "NOT-ENOUGH-INFO (possible Frankenstein-stitched claim)"
    else:
        return "SUPPORTED"


# =======================================================================
# TRACK B: CORRECT (Ponmagal's rewrite.py logic)
# =======================================================================
def generate_correction(claim_text, evidence_nodes, tokenizer, corrector_model):
    evidence_text = "\n".join(f"- {n['sentence_text']}" for n in evidence_nodes)
    prompt = (
        "You are a fact-correction system. You will be given an ORIGINAL CLAIM that "
        "has been flagged as unsupported, along with the ONLY evidence sentences available.\n\n"
        "Your task: rewrite the claim so it states ONLY what the evidence sentences directly "
        "support. Rules:\n"
        "1. Do NOT introduce any causal relationship (because, due to, caused by, led to) "
        "unless one of the evidence sentences explicitly states that relationship.\n"
        "2. If the evidence only supports separate, unconnected facts, rewrite the claim as "
        "separate factual statements instead of a causal claim.\n"
        "3. Do NOT add any fact that is not present in the evidence sentences.\n"
        "4. Keep the rewrite concise - 1 to 2 sentences.\n"
        "5. Output ONLY the corrected rewrite text. No preamble, no explanation.\n\n"
        "ORIGINAL CLAIM:\n"
        f"{claim_text}\n\n"
        "EVIDENCE SENTENCES:\n"
        f"{evidence_text}\n\n"
        "CORRECTED REWRITE:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = corrector_model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# =======================================================================
# FULL INTEGRATED PIPELINE
# =======================================================================
def run_full_pipeline(llm_answer_sentences, evidence_documents, evidence_sources):
    print("Loading models (cached after first run)...")
    nli_classifier = hf_pipeline("text-classification", model="roberta-large-mnli")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    corrector_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    print(f"\n{'='*70}")
    print("STEP A1: EXTRACT - finding causal-looking sentences (Deeksha's logic)")
    print(f"{'='*70}")
    causal_candidates = extract_causal_sentences(llm_answer_sentences)
    for s in causal_candidates:
        print(f"  Found: {s}")

    print(f"\n{'='*70}")
    print("STEP A2: CLASSIFY - confirming true causal claims (Deeksha's logic)")
    print(f"{'='*70}")
    confirmed_claims = []
    for sentence in causal_candidates:
        label = classify_sentence(sentence, nli_classifier)
        print(f"  {label:>13}: {sentence}")
        if label == "causal":
            confirmed_claims.append(sentence)

    print(f"\n{len(confirmed_claims)} confirmed causal claim(s) will be verified against evidence.\n")

    all_results = []
    for claim_text in confirmed_claims:
        print(f"\n{'='*70}")
        print(f"STEP B: VERIFYING CLAIM: {claim_text}")
        print(f"{'='*70}")

        retrieved = retrieve_evidence(claim_text, evidence_documents, evidence_sources, embed_model)
        print("  Retrieved evidence:")
        for r in retrieved:
            print(f"    [{r['source']}] {r['sentence']}")

        evidence_nodes = build_evidence_graph(claim_text, retrieved, nli_classifier)
        verdict = get_verdict(evidence_nodes)
        print(f"\n  VERDICT: {verdict}")

        result = {
            "claim_text": claim_text,
            "evidence_graph": evidence_nodes,
            "verdict": verdict,
        }

        if verdict != "SUPPORTED":
            corrected = generate_correction(claim_text, evidence_nodes, tokenizer, corrector_model)
            print(f"  Corrected: {corrected}")
            result["corrected_claim"] = corrected
        else:
            result["corrected_claim"] = None

        all_results.append(result)

    print(f"\n{'='*70}")
    print("FULL INTEGRATED RESULT (JSON)")
    print(f"{'='*70}")
    print(json.dumps(all_results, indent=2))
    return all_results


# =======================================================================
# TEST DATA
# =======================================================================
if __name__ == "__main__":
    # Simulated "LLM-generated answer" - mix of causal-sounding and plain sentences
    llm_answer_sentences = [
        "Quentin Tarantino directed the Palme d'Or winning film because he was born in Tennessee.",
        "Pulp Fiction is a 1994 American crime film directed by Quentin Tarantino.",
        "The weather was sunny during the Cannes Film Festival.",
        "Tarantino is a British film director due to his unique storytelling style.",
    ]

    # Real evidence documents (same as Ponmagal's earlier tests)
    titles = ["Pulp Fiction", "Quentin Tarantino"]
    sentences_per_doc = [
        [
            "Pulp Fiction is a 1994 American crime film written and directed by Quentin Tarantino.",
            "The film won the Palme d'Or at the 1994 Cannes Film Festival.",
            "It stars John Travolta, Samuel L. Jackson, and Uma Thurman.",
        ],
        [
            "Quentin Jerome Tarantino is an American film director, screenwriter, and producer.",
            "He was born on March 27, 1963, in Knoxville, Tennessee.",
            "Tarantino is known for his nonlinear storylines and stylized violence.",
        ],
    ]

    evidence_documents = []
    evidence_sources = []
    for title, sentences in zip(titles, sentences_per_doc):
        for sentence in sentences:
            evidence_documents.append(sentence.strip())
            evidence_sources.append(title)

    run_full_pipeline(llm_answer_sentences, evidence_documents, evidence_sources)