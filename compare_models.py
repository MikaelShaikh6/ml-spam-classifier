
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import pipeline

# Load the same dataset
df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["Category", "Message"],
    encoding="utf-8"
)

X = df["Message"]
y = df["Category"].str.strip().str.lower()

# Reproduce the baseline's exact test split
_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load the pretrained transformer
print("Loading pretrained model (first run downloads it)...")
classifier = pipeline(
    "text-classification",
    model="didulantha/sms-spam-detector",
    device=-1
)

# Predict on the held-out messages
messages = X_test.tolist()
start = time.perf_counter()

results = []
for i in range(0, len(messages), 16):
    results.extend(
        classifier(
            messages[i:i + 16],
            truncation=True,
            max_length=128
        )
    )

elapsed = time.perf_counter() - start

# Normalize model labels to ham/spam
LABEL_MAP = {
    "label_0": "ham",   # Verify this
    "label_1": "spam",  # Verify this
}

predictions = []
for result in results:
    label = result["label"].lower()
    if label not in LABEL_MAP:
        raise ValueError(f"Unexpected model label: {label}")
    predictions.append(LABEL_MAP[label])

print("\nTransformer evaluation:")
print(classification_report(y_test, predictions, digits=4))
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(f"Total inference time: {elapsed:.2f} seconds")
print(f"Average per message: {elapsed / len(messages) * 1000:.2f} ms")
