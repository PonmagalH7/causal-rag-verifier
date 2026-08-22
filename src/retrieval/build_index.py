"""
Track B - Step 1: Prove retrieval works end to end.

Takes a small set of documents (simulating a HotpotQA-style multi-source
example), embeds them, builds a FAISS index, and retrieves the most
relevant documents for a query.

Run: python src/retrieval/build_index.py
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------------------------------------------------------------------
# 1. Sample multi-source documents (stand-in for a real HotpotQA example)
#    Later this will be loaded from data/raw/hotpotqa/
# ---------------------------------------------------------------------
documents = [
    "The dam near Riverside was built in 1965 and had not been inspected since 2010.",
    "Heavy rainfall in the region caused water levels to rise sharply over three days.",
    "The village of Millbrook is located five kilometers downstream from the dam.",
    "Local authorities issued an evacuation notice after cracks were spotted in the dam wall.",
    "The flood destroyed most of the homes in Millbrook within a few hours.",
    "Engineers later confirmed the dam collapse was caused by structural fatigue, not rainfall alone.",
]

query = "Why did the village get destroyed?"

# ---------------------------------------------------------------------
# 2. Embed documents using a small, fast sentence-transformer model
# ---------------------------------------------------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding documents...")
doc_embeddings = model.encode(documents, convert_to_numpy=True)

# ---------------------------------------------------------------------
# 3. Build a FAISS index (flat L2 index - simplest possible, fine for now)
# ---------------------------------------------------------------------
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)
print(f"Indexed {index.ntotal} documents (embedding dim = {dimension})")

# ---------------------------------------------------------------------
# 4. Embed the query and retrieve top-k most relevant documents
# ---------------------------------------------------------------------
top_k = 3
query_embedding = model.encode([query], convert_to_numpy=True)
distances, indices = index.search(query_embedding, top_k)

# ---------------------------------------------------------------------
# 5. Print results
# ---------------------------------------------------------------------
print(f"\nQuery: {query}\n")
print(f"Top {top_k} retrieved documents:")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
    print(f"{rank}. (distance={dist:.4f}) {documents[idx]}")