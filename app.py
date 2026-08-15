
import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline # Import Pipeline to correctly load the pipeline object

# Ensure clean_text function is available in app.py
def clean_text(text):
    """Advanced normalization: lowercasing, punctuation removal, and whitespace collapse."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Ensure NYSCFAQChatbot class is available in app.py
class NYSCFAQChatbot:
    def __init__(self, dataset, vectorizer, classifier, fallback_threshold=0.3):
        self.df = dataset.drop_duplicates(subset=['clean_question']).reset_index(drop=True)
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.fallback_threshold = fallback_threshold
        self.corpus_tfidf = self.vectorizer.transform(self.df['clean_question'])

    def respond(self, user_query):
        cleaned_query = clean_text(user_query)
        if not cleaned_query:
            return self._fallback_response(user_query, 0.0)

        query_vec = self.vectorizer.transform([cleaned_query])
        probs = self.classifier.predict_proba(query_vec)[0]
        max_prob_idx = np.argmax(probs)
        predicted_intent = self.classifier.classes_[max_prob_idx]
        intent_confidence = float(probs[max_prob_idx])

        sim_scores = cosine_similarity(query_vec, self.corpus_tfidf).flatten()
        best_match_idx = np.argmax(sim_scores)
        similarity_score = float(sim_scores[best_match_idx])

        if similarity_score < self.fallback_threshold:
             return self._fallback_response(user_query, similarity_score)

        matched_row = self.df.iloc[best_match_idx]

        return {
            "query": user_query,
            "intent": matched_row['intent'],
            "category": matched_row.get('category', 'General'),
            "matched_faq": matched_row['question'],
            "answer": matched_row['answer'],
            "confidence": round((intent_confidence + similarity_score) / 2, 3),
            "similarity_score": round(similarity_score, 3),
            "is_fallback": False
        }

    def _fallback_response(self, user_query, confidence):
        return {
            "query": user_query,
            "intent": "fallback",
            "category": "Uncertain",
            "matched_faq": None,
            "answer": "I'm sorry, I couldn't find a specific answer to that. Could you please rephrase or ask about NYSC registration, relocation, or allowances?",
            "confidence": round(confidence, 3),
            "similarity_score": 0.0,
            "is_fallback": True
        }

# Streamlit app logic
st.title("🇳🇬 NYSC Assistant")

# Load data and models
df = pd.read_excel('/content/nysc_faq_dataset.xlsx')
df['clean_question'] = df['question'].apply(clean_text) # Use the clean_text defined above

# Load the entire pipeline, then extract vectorizer and classifier
pipeline = joblib.load('nysc_pipeline.joblib')
vectorizer = pipeline.named_steps['tfidf']
classifier = pipeline.named_steps['classifier']

# Initialize chatbot
bot = NYSCFAQChatbot(df, vectorizer, classifier)

query = st.text_input("How can I help you today?")
if query:
    res = bot.respond(query)
    if res['is_fallback']:
        st.warning(res['answer'])
    else:
        st.success(res['answer'])
        st.caption(f"Source FAQ: {res['matched_faq']}")
