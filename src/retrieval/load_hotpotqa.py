"""
Track B - Step 2: Test retrieval on a REAL HotpotQA example.

Downloads a small slice of the HotpotQA dataset (via HuggingFace datasets),
picks one real multi-document question, and runs it through the same
FAISS retrieval pipeline proven in build_index.py.

Run: python src/retrieval/load_hotpotqa.py
"""

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------------------------------------------------------------------
# 1. Load a small slice of HotpotQA (distractor split, validation set)
#    This downloads only what's needed, not the full dataset.
# ---------------------------------------------------------------------
print("Downloading a small slice of HotpotQA...")
dataset = load_dataset("hotpot_qa", "distractor", split="validation[:1]", trust_remote_code=True)

example = dataset[0]

question = example["question"]
answer = example["answer"]

# HotpotQA stores supporting documents as a dict of titles + sentence lists
titles = example["context"]["title"]
sentences_per_doc = example["context"]["sentences"]

# Flatten all sentences from all documents into one list of passages
documents = []
for title, sentences in zip(titles, sentences_per_doc):
    for sentence in sentences:
        documents.append(f"[{title}] {sentence.strip()}")

print(f"\nQuestion: {question}")
print(f"Ground-truth answer: {answer}")
print(f"Total sentences retrieved from {len(titles)} source documents: {len(documents)}")

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