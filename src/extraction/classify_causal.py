from transformers import pipeline

# Load the NLI model
classifier = pipeline(
    "text-classification",
    model="roberta-large-mnli"
)

# Test sentences
sentences = [
    "The dam collapsed because of structural fatigue.",
    "The accident resulted in a traffic jam.",
    "The dam collapsed and the village flooded.",
    "The rain stopped and then the road became dry."
]

# Temporal words
temporal_markers = [
    "then",
    "after",
    "before",
    "later",
    "followed by",
    "subsequently"
]

# Causal hypothesis
causal_hypothesis = "This sentence states that one event caused another event."

for sentence in sentences:

    # Check for obvious temporal relationships first
    sentence_lower = sentence.lower()

    if any(marker in sentence_lower for marker in temporal_markers):
        label = "temporal"

    else:
        # NLI checks whether the causal interpretation is supported
        result = classifier(
            f"{sentence} </s></s> {causal_hypothesis}"
        )[0]

        if result["label"] == "ENTAILMENT":
            label = "causal"
        else:
            label = "correlational"

    print(f"{label}: {sentence}")