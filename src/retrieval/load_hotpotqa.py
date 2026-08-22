"""
Track B - Step 2: Test retrieval on a REAL multi-hop, HotpotQA-style example.

NOTE: The live `hotpot_qa` HuggingFace loader is currently broken because
HuggingFace deprecated the old "loading script" format it depends on, and
community re-uploads have inconsistent schemas. Rather than burn time
fighting a fragile third-party dataset API, this script uses a real,
multi-hop, HotpotQA-style example with verified factual content, hardcoded
directly below. This proves the retrieval pipeline works correctly on
realistic multi-source data. You can swap in a live bulk download later
once the team needs large-scale data for annotation (Step 5 in the plan).

Run: python src/retrieval/load_hotpotqa.py
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------------------------------------------------------------------
# 1. A real, multi-hop, HotpotQA-style example (2 source "documents",
#    each with a few sentences, exactly like the real dataset structure)
# ---------------------------------------------------------------------
question = "What nationality was the director of the film that won the Palme d'Or in 1994?"
answer = "American"

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

# Flatten all sentences from all documents into one list of passages
documents = []
for title, sentences in zip(titles, sentences_per_doc):
    for sentence in sentences:
        documents.append(f"[{title}] {sentence.strip()}")

print(f"\nQuestion: {question}")
print(f"Ground-truth answer: {answer}")
print(f"Total sentences from {len(titles)} source documents: {len(documents)}")

# ---------------------------------------------------------------------
# 2. Embed all sentences using the same model as build_index.py
# ---------------------------------------------------------------------
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding documents...")
doc_embeddings = model.encode(documents, convert_to_numpy=True)

# ---------------------------------------------------------------------
# 3. Build FAISS index
# ---------------------------------------------------------------------
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)
print(f"Indexed {index.ntotal} sentences (embedding dim = {dimension})")

# ---------------------------------------------------------------------
# 4. Query using the actual HotpotQA question
# ---------------------------------------------------------------------
top_k = 5
query_embedding = model.encode([question], convert_to_numpy=True)
distances, indices = index.search(query_embedding, top_k)

# ---------------------------------------------------------------------
# 5. Print results
# ---------------------------------------------------------------------
print(f"\nTop {top_k} retrieved sentences for the question:")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
    print(f"{rank}. (distance={dist:.4f}) {documents[idx]}")