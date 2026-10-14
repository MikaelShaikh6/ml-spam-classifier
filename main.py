from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pandas as pd

# Load the downloaded SMS spam dataset
df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["Category", "Message"],
    encoding="utf-8"
)

X = df["Message"]
y = df["Category"].str.strip().str.lower()

print(f"Loaded {len(df)} messages")
print(df["Category"].value_counts())

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create the ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 4. Train the model
model.fit(X_train, y_train)

# 5. Evaluate on messages the model hasn't seen
predictions = model.predict(X_test)

print("Model evaluation:")
print(classification_report(y_test, predictions))

# 6. Display the confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions,
    labels=["ham", "spam"]
)
plt.title("SMS Spam Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix.png")

# 7. Try your own messages
examples = [
    "Congratulations! You won a free prize. Click now!",
    "Hey, are we still meeting for lunch tomorrow?"
]

for message, prediction in zip(examples, model.predict(examples)):
    print(f"\nMessage: {message}")
    print(f"Prediction: {prediction}")
