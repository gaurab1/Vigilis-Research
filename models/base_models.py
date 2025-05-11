import pandas as pd
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load and preprocess datasets
email_ds = pd.read_csv('data/spamassassin.csv')
text_ds = pd.read_csv('data/spam.csv', encoding='latin-1')

# Shuffle the datasets
text_ds = text_ds.sample(frac=1, random_state=42)[['label', 'text']]
email_ds = email_ds.sample(frac=1, random_state=42)[['label', 'text']]

def train_evaluate_model(X_train, X_test, y_train, y_test, model_name, model):
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    spam_recall = report['0']['recall']
    
    print(f"\n{model_name} Results:")
    print(f"Overal Accuracy: {accuracy:.4f}")
    print(f"% of Spam Caught: {spam_recall:.4f}")
    
    return model

def evaluate_model(model, X, y, featurizer):
    X_transform= featurizer.transform(X)
    y_pred = model.predict(X_transform)

    report = classification_report(y, y_pred, output_dict=True)
    spam_recall = report['0']['recall']
    print("Accuracy:", accuracy_score(y, y_pred))
    print("% of Spam caught:", spam_recall)
    print("\n")

# Do some dataset preprocessing
X_text = text_ds['text']
y_text = text_ds['label']

X_email = email_ds['text']
y_email = email_ds['label']

# Split the data
X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(
    X_text, y_text, test_size=0.2, random_state=42
)

X_email_train, X_email_test, y_email_train, y_email_test = train_test_split(
    X_email, y_email, test_size=0.2, random_state=42
)

# Train models for text messages
print("Training models for text messages...")

# Create TF-IDF features
tfidf_text = TfidfVectorizer(max_features=5000)
X_text_train_tfidf = tfidf_text.fit_transform(X_text_train)
X_text_test_tfidf = tfidf_text.transform(X_text_test)

# Train and evaluate models for text messages
nb_text = train_evaluate_model(
    X_text_train_tfidf, X_text_test_tfidf, 
    y_text_train, y_text_test,
    "Naive Bayes (Text Messages)",
    MultinomialNB()
)

lr_text = train_evaluate_model(
    X_text_train_tfidf, X_text_test_tfidf, 
    y_text_train, y_text_test,
    "Logistic Regression (Text Messages)",
    LogisticRegression(max_iter=10000, class_weight='balanced')
)

rf_text = train_evaluate_model(
    X_text_train_tfidf, X_text_test_tfidf, 
    y_text_train, y_text_test,
    "Random Forest (Text Messages)",
    RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    )
)

hgbt_text = train_evaluate_model(
    X_text_train_tfidf.toarray(), X_text_test_tfidf.toarray(), 
    y_text_train, y_text_test,
    "HistGradientBoosting (Text Messages)",
    HistGradientBoostingClassifier(max_iter=100, class_weight='balanced')
)

# Train models for emails
print("\nTraining models for emails...")

# Create TF-IDF features
tfidf_email = TfidfVectorizer(max_features=5000)
X_email_train_tfidf = tfidf_email.fit_transform(X_email_train)
X_email_test_tfidf = tfidf_email.transform(X_email_test)

# Train and evaluate models for emails
nb_email = train_evaluate_model(
    X_email_train_tfidf, X_email_test_tfidf, 
    y_email_train, y_email_test,
    "Naive Bayes (Emails)",
    MultinomialNB()
)

lr_email = train_evaluate_model(
    X_email_train_tfidf, X_email_test_tfidf, 
    y_email_train, y_email_test,
    "Logistic Regression (Emails)",
    LogisticRegression(max_iter=10000, class_weight='balanced')
)

rf_email = train_evaluate_model(
    X_email_train_tfidf, X_email_test_tfidf, 
    y_email_train, y_email_test,
    "Random Forest (Emails)",
    RandomForestClassifier(class_weight='balanced')
)

hgbt_email = train_evaluate_model(
    X_email_train_tfidf.toarray(), X_email_test_tfidf.toarray(), 
    y_email_train, y_email_test,
    "HistGradientBoosting (Emails)",
    HistGradientBoostingClassifier(class_weight='balanced')
)

# Combine the datasets
email_ds['type'] = 'email'
text_ds['type'] = 'text'
combined_ds = pd.concat([email_ds, text_ds], ignore_index=True)
combined_ds = combined_ds.sample(frac=1, random_state=42)  # Shuffle the combined dataset

X_combined = combined_ds['text']
y_combined = combined_ds['label']

# Split the combined data
# X_combined_train, X_combined_test, y_combined_train, y_combined_test = train_test_split(
#     X_combined, y_combined, test_size=0.2, random_state=42
# )
test_set = combined_ds.iloc[:int(0.15 * len(combined_ds))]
train_set = combined_ds.iloc[int(0.15 * len(combined_ds)):]

# Create TF-IDF features for combined data
tfidf_combined = TfidfVectorizer(max_features=5000)
X_combined_train_tfidf = tfidf_combined.fit_transform(train_set['text'])
X_combined_test_tfidf = tfidf_combined.transform(test_set['text'])

# Train and evaluate models on combined data
print("\nTraining models on combined dataset (emails + text messages)...")

nb_combined = train_evaluate_model(
    X_combined_train_tfidf, X_combined_test_tfidf, 
    train_set['label'], test_set['label'],
    "Naive Bayes (Combined Dataset)",
    MultinomialNB()
)

lr_combined = train_evaluate_model(
    X_combined_train_tfidf, X_combined_test_tfidf, 
    train_set['label'], test_set['label'],
    "Logistic Regression (Combined Dataset)",
    LogisticRegression(max_iter=10000, class_weight='balanced')
)

rf_combined = train_evaluate_model(
    X_combined_train_tfidf, X_combined_test_tfidf, 
    train_set['label'], test_set['label'],
    "Random Forest (Combined Dataset)",
    RandomForestClassifier(class_weight='balanced')
)

gbt_combined = train_evaluate_model(
    X_combined_train_tfidf.toarray(), X_combined_test_tfidf.toarray(), 
    train_set['label'], test_set['label'],
    "Gradient Boosting (Combined Dataset)",
    HistGradientBoostingClassifier(max_iter=100, class_weight='balanced')
)

test_set['prediction'] = lr_combined.predict(X_combined_test_tfidf)

print("Total samples: ", len(combined_ds))
print("Email samples: ", len(email_ds))
print("Text message samples: ", len(text_ds))

print("Spam ratio in combined dataset: ", combined_ds['label'].mean())
print("Spam ratio in email dataset: ", email_ds['label'].mean())
print("Spam ratio in text dataset: ", text_ds['label'].mean())
