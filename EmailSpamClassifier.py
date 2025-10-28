import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

class SpamClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()
    
    def train(self, emails, labels):
        # Convert text to feature vectors
        X = self.vectorizer.fit_transform(emails)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Test accuracy
        accuracy = self.classifier.score(X_test, y_test)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    def predict(self, email):
        X = self.vectorizer.transform([email])
        prediction = self.classifier.predict(X)[0]
        probability = self.classifier.predict_proba(X)[0]
        
        return "Spam" if prediction == 1 else "Not Spam", probability


emails = [
    "Win a free iPhone now! Click here!",
    "Meeting scheduled for tomorrow at 3pm",
    "Congratulations! You've won $1,000,000",
    "Can you send me the project report?",
    "Limited time offer - Buy now!",
    "Let's catch up for coffee next week",
    "URGENT: Your account has been suspended",
    "Thanks for your help with the presentation"
]

labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam


classifier = SpamClassifier()
classifier.train(emails, labels)


test_email = "Congratulations! You won a lottery"
result, prob = classifier.predict(test_email)
print(f"\nEmail: {test_email}")
print(f"Prediction: {result}")
print(f"Confidence: {max(prob) * 100:.2f}%")
