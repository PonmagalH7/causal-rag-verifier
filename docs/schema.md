# Track A ↔ Track B Interface Schema

This is the JSON contract between causal extraction/verification (Track A)
and evidence graph/retrieval (Track B). Any change to this file must be
agreed by both team members before implementation changes.

## Claim Object
```json
{
  "claim_id": "string (unique)",
  "query_id": "string",
  "claim_text": "string",
  "source_ids": ["doc_1", "doc_2"],
  "causal_type": "causal | correlational | temporal",
  "label": "Supported | Correlational-only | Contradicted | Frankenstein-stitched | Not-Enough-Info",
  "evidence_spans": ["exact text span from source"],
  "confidence_score": 0.0,
  "explanation": "string",
  "corrected_claim": "string or null"
}
```

## Evidence Graph Node
```json
{
  "node_id": "string",
  "source_doc": "string",
  "sentence_text": "string",
  "relation_to_claim": "supports | contradicts | unrelated | correlational_only"
}
```
