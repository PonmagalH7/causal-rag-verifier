"""
Track B - Step 4: Correction / Rewrite Module (FREE, local model version).

Uses a free, local, open-source instruction-following model (FLAN-T5)
via HuggingFace transformers - no API key, no cost, runs on CPU.

Run: python src/correction/rewrite.py
"""

import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

claim_object = {
    "claim_id": "claim_001",
    "claim_text": "Quentin Tarantino directed the Palme d'Or winning film because he was born in Tennessee.",
    "source_ids": ["Pulp Fiction", "Quentin Tarantino"],
    "evidence_graph": [
        {
            "node_id": "node_1",
            "source_doc": "Quentin Tarantino",
            "sentence_text": "Quentin Jerome Tarantino is an American film director, screenwriter, and producer.",
            "relation_to_claim": "unrelated",
        },
        {
            "node_id": "node_2",
            "source_doc": "Pulp Fiction",
            "sentence_text": "The film won the Palme d'Or at the 1994 Cannes Film Festival.",
            "relation_to_claim": "unrelated",
        },
        {
            "node_id": "node_3",
            "source_doc": "Quentin Tarantino",
            "sentence_text": "He was born on March 27, 1963, in Knoxville, Tennessee.",
            "relation_to_claim": "unrelated",
        },
        {
            "node_id": "node_4",
            "source_doc": "Pulp Fiction",
            "sentence_text": "Pulp Fiction is a 1994 American crime film written and directed by Quentin Tarantino.",
            "relation_to_claim": "unrelated",
        },
    ],
    "verdict": "NOT-ENOUGH-INFO / possible Frankenstein-stitched claim",
}

evidence_text = "\n".join(
    f"- {node['sentence_text']}" for node in claim_object["evidence_graph"]
)

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
    f"{claim_object['claim_text']}\n\n"
    "EVIDENCE SENTENCES:\n"
    f"{evidence_text}\n\n"
    "CORRECTED REWRITE:"
)

print("Original claim (flagged):")
print(f"  {claim_object['claim_text']}\n")
print(f"Verdict from evidence graph: {claim_object['verdict']}\n")

print("Loading local correction model (first run downloads ~1GB, then cached)...")
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Generating evidence-consistent rewrite...\n")
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
output_ids = model.generate(**inputs, max_new_tokens=100)
corrected_claim = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

print("Corrected rewrite:")
print(f"  {corrected_claim}\n")

result = {
    **claim_object,
    "corrected_claim": corrected_claim,
}

print("--- Full Result ---")
print(json.dumps(result, indent=2))
