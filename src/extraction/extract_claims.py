import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Test sentences
sentences = [
    "The dam collapsed because of structural fatigue.",
    "The weather was sunny today.",
    "The accident resulted in a traffic jam.",
    "The student studied for three hours.",
    "The flood was caused by heavy rainfall."
]

# Causal discourse markers
causal_markers = [
    "because",
    "due to",
    "caused by",
    "led to",
    "as a result of",
    "resulted in"
]

# Check each sentence
for sentence in sentences:
    doc = nlp(sentence)

    sentence_lower = sentence.lower()

    if any(marker in sentence_lower for marker in causal_markers):
        print(sentence)

