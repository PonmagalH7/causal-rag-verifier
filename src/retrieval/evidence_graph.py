"""
Track B - Step 3: Build a basic Evidence Graph.

Takes retrieved sentences (from load_hotpotqa.py / build_index.py) and a
causal claim, then uses an NLI (Natural Language Inference) model to
classify each sentence's relationship to the claim:
    - "supports"    -> sentence entails the claim
    - "contradicts" -> sentence contradicts the claim
    - "unrelated"   -> sentence is neutral / not relevant

This is the first real piece of the evidence graph structure defined in
docs/schema.md. Later, Track A's causal-claim extractor will feed claims
into this module automatically instead of the hardcoded claim below.

Run: python src/retrieval/evidence_graph.py
"""

from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import json

# ---------------------------------------------------------------------
# 1. Same real multi-hop example as load_hotpotqa.py
# ---------------------------------------------------------------------
question = "What nationality was the director of the film that won the Palme d'Or in 1994?"

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

# ---------------------------------------------------------------------
# 2. The claim we want to verify against the evidence
#    (later this comes from Track A's causal-claim extractor)
# ---------------------------------------------------------------------
claim_text = "Quentin Tarantino directed the Palme d'Or winning film because he was born in Tennessee."
claim_id = "claim_001"

print(f"Claim to verify: {claim_text}\n")

# ---------------------------------------------------------------------
# 3. Retrieve the most relevant sentences for this claim (reuse FAISS)
# ---------------------------------------------------------------------
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = embed_model.encode(documents, convert_to_numpy=True)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

top_k = 5
claim_embedding = embed_model.encode([claim_text], convert_to_numpy=True)
distances, indices = index.search(claim_embedding, top_k)

# ---------------------------------------------------------------------
# 4. Load an NLI model to classify each sentence vs. the claim
#    roberta-large-mnli outputs: CONTRADICTION, NEUTRAL, ENTAILMENT
# ---------------------------------------------------------------------
print("Loading NLI model (this may take a minute on first run)...")
nli = pipeline("text-classification", model="roberta-large-mnli")

label_map = {
    "ENTAILMENT": "supports",
    "CONTRADICTION": "contradicts",
    "NEUTRAL": "unrelated",
}

# ---------------------------------------------------------------------
# 5. Build the evidence graph nodes
# ---------------------------------------------------------------------
evidence_nodes = []
for rank, idx in enumerate(indices[0]):
    sentence = documents[idx]
    source = source_titles[idx]

    # NLI models expect (premise, hypothesis) - premise = evidence, hypothesis = claim
    result = nli(f"{sentence}</s></s>{claim_text}")[0]
    relation = label_map.get(result["label"], "unrelated")

    node = {
        "node_id": f"node_{rank+1}",
        "source_doc": source,
        "sentence_text": sentence,
        "relation_to_claim": relation,
        "nli_confidence": round(result["score"], 4),
    }
    evidence_nodes.append(node)

# ---------------------------------------------------------------------
# 6. Assemble the full claim object (matches docs/schema.md)
# ---------------------------------------------------------------------
claim_object = {
    "claim_id": claim_id,
    "claim_text": claim_text,
    "source_ids": list(set(source_titles)),
    "evidence_graph": evidence_nodes,
}

print("\n--- Evidence Graph ---\n")
print(json.dumps(claim_object, indent=2))

# ---------------------------------------------------------------------
# 7. Quick verdict based on the evidence graph
# ---------------------------------------------------------------------
supports = [n for n in evidence_nodes if n["relation_to_claim"] == "supports"]
contradicts = [n for n in evidence_nodes if n["relation_to_claim"] == "contradicts"]

print("\n--- Verdict ---")
if contradicts:
    print("CONTRADICTED: at least one source directly contradicts this claim.")
elif not supports:
    print("NOT-ENOUGH-INFO: no single source directly supports this causal claim.")
    print("This may indicate a FRANKENSTEIN-STITCHED claim - each piece is true")
    print("individually, but the full causal link is not supported by any source.")
else:
    print("SUPPORTED: at least one source directly entails this claim.")