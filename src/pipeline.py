"""
Track B - Step 5: Unified End-to-End Pipeline.

Chains together everything built so far into one working system:
    1. RETRIEVE  - find the most relevant evidence sentences for a claim (FAISS)
    2. VERIFY    - classify each sentence relation to the claim (NLI)
    3. CORRECT   - if unsupported, generate an evidence-consistent rewrite (FLAN-T5)

Run: python src/pipeline.py
"""

import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import faiss
import numpy as np


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


def run_pipeline(claim_text, documents, source_titles):
    print(f"\n{'='*70}")
    print(f"CLAIM: {claim_text}")
    print(f"{'='*70}\n")

    print("[1/3] Loading models (cached after first run)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    nli_model = pipeline("text-classification", model="roberta-large-mnli")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    corrector_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    print("[1/3] RETRIEVE - finding relevant evidence...")
    retrieved = retrieve_evidence(claim_text, documents, source_titles, embed_model)
    for r in retrieved:
        print(f"    [{r['source']}] {r['sentence']}")

    print("\n[2/3] VERIFY - building evidence graph with NLI...")
    evidence_nodes = build_evidence_graph(claim_text, retrieved, nli_model)
    verdict = get_verdict(evidence_nodes)
    for n in evidence_nodes:
        print(f"    {n['relation_to_claim']:>12} ({n['nli_confidence']:.2f})  {n['sentence_text']}")
    print(f"\n    VERDICT: {verdict}")

    result = {
        "claim_text": claim_text,
        "retrieved_evidence": retrieved,
        "evidence_graph": evidence_nodes,
        "verdict": verdict,
    }

    if verdict != "SUPPORTED":
        print("\n[3/3] CORRECT - generating evidence-consistent rewrite...")
        corrected = generate_correction(claim_text, evidence_nodes, tokenizer, corrector_model)
        print(f"    Original:  {claim_text}")
        print(f"    Corrected: {corrected}")
        result["corrected_claim"] = corrected
    else:
        print("\n[3/3] CORRECT - skipped, claim is already supported.")
        result["corrected_claim"] = None

    print(f"\n{'='*70}")
    print("FULL RESULT (JSON)")
    print(f"{'='*70}")
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
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

    documents = []
    source_titles = []
    for title, sentences in zip(titles, sentences_per_doc):
        for sentence in sentences:
            documents.append(sentence.strip())
            source_titles.append(title)

        test_claims = [
        # 1. Frankenstein-stitched: both halves true, but the causal link is not.
        "Quentin Tarantino directed the Palme d'Or winning film because he was born in Tennessee.",

        # 2. Directly SUPPORTED: stated word-for-word in one source sentence.
        "Pulp Fiction is a 1994 American crime film directed by Quentin Tarantino.",

        # 3. CONTRADICTED: claims a different nationality than the source states.
        "Quentin Tarantino is a British film director.",
    ]

    for claim in test_claims:
        run_pipeline(claim, documents, source_titles)