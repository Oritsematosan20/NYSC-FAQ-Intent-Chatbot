# 🇳🇬 NYSC FAQ Intent Chatbot

> **An AI-powered Natural Language Processing chatbot that helps Nigerian National Youth Service Corps (NYSC) members get quick answers to frequently asked questions.**


## 📌 Project Overview

The **NYSC FAQ Intent Chatbot** is an AI/ML application designed to provide fast and relevant answers to common questions asked by prospective and serving National Youth Service Corps members.

The project addresses a practical problem: corps members frequently ask similar questions concerning **registration, deployment, orientation camp, relocation, clearance, allowances, certificates, and other NYSC-related processes**.

Instead of requiring users to manually search through large amounts of information, the chatbot accepts a natural-language question, identifies the most likely intent, finds the most relevant FAQ, and returns the corresponding answer.

The original project objective was to develop an efficient and accessible AI-powered tool capable of assisting corps members and other NYSC stakeholders with common inquiries.


## 🎯 Problem Statement

NYSC participants encounter numerous administrative and procedural questions throughout their service year.

Common questions include:

* How do I register for NYSC?
* How do I know my deployment state?
* What happens during orientation camp?
* How does monthly clearance work?
* When is NYSC allowance paid?
* Can I relocate after deployment?
* How do I obtain my NYSC certificate?
* What happens if I miss clearance?
* What is SAED?
* What is the NYSC program?

Repeatedly answering these questions manually can be time-consuming.

This project demonstrates how **machine learning and natural language processing** can be used to build a lightweight FAQ assistant capable of automatically classifying user questions and retrieving appropriate answers.


## 💡 Solution

The chatbot combines two complementary NLP techniques:

1. **Intent Classification**

   * Converts user questions into TF-IDF numerical representations.
   * Uses a trained machine-learning classifier to determine the likely NYSC intent.

2. **FAQ Semantic Matching**

   * Compares the user's question with the cleaned FAQ corpus using cosine similarity.
   * Selects the most relevant FAQ record.
   * Returns the corresponding answer.

A confidence threshold is also implemented so that uncertain queries are not blindly answered. Instead, the chatbot provides a fallback response asking the user to rephrase the question.

The chatbot engine implements this combination of classification, similarity matching, and fallback handling.


# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │     User Query      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Text Cleaning     │
                    │ Lowercase            │
                    │ Remove punctuation   │
                    │ Normalize whitespace │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   TF-IDF Vectorizer │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌──────────────────┐       ┌────────────────────┐
       │ Intent Classifier│       │ Cosine Similarity  │
       │     SVM          │       │ FAQ Matching       │
       └────────┬─────────┘       └──────────┬─────────┘
                │                            │
                └────────────┬───────────────┘
                             ▼
                 ┌────────────────────────┐
                 │ Confidence / Threshold │
                 │       Evaluation       │
                 └───────────┬────────────┘
                             │
                    ┌────────┴─────────┐
                    │                  │
                    ▼                  ▼
             ┌─────────────┐    ┌─────────────┐
             │ FAQ Answer  │    │  Fallback   │
             └─────────────┘    └─────────────┘
```


# 🧠 Machine Learning Pipeline

The project uses a Scikit-learn pipeline containing:

```text
Raw Question
     ↓
Text Cleaning
     ↓
TF-IDF Vectorization
     ↓
SVM Classifier
     ↓
Predicted Intent
```

The trained pipeline combines the TF-IDF vectorizer and selected classifier into a single reusable object and is saved as `nysc_pipeline.joblib`. This ensures that the same vectorization process used during training is available during inference.


# 📊 Dataset

The project uses an NYSC FAQ dataset containing questions, answers, intents, and categories.

### Dataset statistics

| Property                        | Value |
| ------------------------------- | ----: |
| Original records                |   824 |
| Records after duplicate removal |   822 |
| Question records                |   822 |
| Unique answers                  |   760 |
| Intent classes                  |    12 |
| Category classes                |    13 |

The notebook explicitly reports 824 initial records and 822 unique FAQ records after cleaning.

The dataset contains fields including:

```text
question
answer
intent
category
clean_question
```

Examples include questions concerning allowances, clearance, registration, and other NYSC-related topics.


# 🧹 Data Preprocessing

Before training, the questions undergo text normalization.

The preprocessing pipeline performs:

* Conversion to lowercase
* Removal of punctuation and special characters
* Removal/replacement of unsupported characters
* Whitespace normalization
* Duplicate-question removal

Example:

```text
Original:
"How much is NYSC monthly allowance?"

Cleaned:
"how much is nysc monthly allowance"
```

The notebook implements this normalization using regular expressions and subsequently removes duplicate cleaned questions.


# 🔎 TF-IDF Feature Engineering

The chatbot uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert natural-language questions into numerical feature vectors.

TF-IDF enables the machine-learning model to identify terms that are important for distinguishing between different NYSC intents.

The notebook imports and uses Scikit-learn's `TfidfVectorizer` as part of the NLP pipeline.


# 🤖 Model Selection

Several traditional machine-learning approaches were considered during model development, including:

* Logistic Regression
* Linear Support Vector Machine
* Multinomial Naive Bayes

The final implementation uses a **tuned SVM classifier** as the selected model.

The notebook specifically constructs the final pipeline using the best SVM model.

## For comparison, the notebook reports a mean 5-fold cross-validation accuracy of approximately **70.32% for the selected Logistic Regression configuration**, while the selected classifier achieved approximately **89.65% mean cross-validation accuracy**.

# 📈 Model Performance

The final SVM classifier was evaluated on a held-out test set.

### Evaluation Results

| Metric             |      Score |
| ------------------ | ---------: |
| Accuracy           | **90.30%** |
| Macro Precision    |    **94%** |
| Macro Recall       |    **86%** |
| Macro F1-score     |    **89%** |
| Weighted Precision |    **91%** |
| Weighted Recall    |    **90%** |
| Weighted F1-score  |    **90%** |

The notebook's classification report shows an overall test accuracy of **90.30%** across 165 test samples.

A confusion matrix was also generated to inspect classification performance across the different intent classes.


# 💬 Chatbot Engine

The `NYSCFAQChatbot` class is responsible for converting the trained model into a usable question-answering engine.

For every query, the chatbot:

1. Cleans the user's input.
2. Converts the query into a TF-IDF vector.
3. Calculates intent probabilities.
4. Identifies the most probable intent.
5. Calculates cosine similarity against the FAQ corpus.
6. Finds the most relevant FAQ.
7. Combines classification confidence and similarity.
8. Returns the answer when the similarity is sufficiently high.
9. Otherwise, returns a fallback response.

The implementation pre-computes the TF-IDF representation of the FAQ corpus to support similarity matching during inference.


# 🛡️ Fallback Mechanism

A key feature of this project is its **fallback mechanism**.

Rather than providing an answer for every query regardless of confidence, the chatbot checks the similarity score against a predefined threshold.

The deployed chatbot configuration uses a fallback threshold of **0.50**.

For example:

```text
User:
Can I Japa?

Chatbot:
I'm sorry, I couldn't find a specific answer to that.
Could you please rephrase or ask about NYSC registration,
relocation, or allowances?
```

This behaviour was demonstrated during interactive testing, where an unrelated query was correctly routed to the fallback response.


# 🖥️ Streamlit Application

The chatbot is deployed through a **Streamlit web interface**.

The application provides a simple input box where users can enter questions and receive answers.

The Streamlit application:

* Loads the FAQ dataset.
* Loads the trained `nysc_pipeline.joblib`.
* Extracts the TF-IDF vectorizer and classifier.
* Initializes the chatbot engine.
* Accepts user questions.
* Displays the generated answer.
* Displays the matched FAQ when a valid match is found.
* Displays a warning when the chatbot falls back.

These behaviours are implemented directly in `app.py`.


# 📁 Project Structure

```text
NYSC-FAQ-Intent-Chatbot/
│
├── app.py
├── nysc_pipeline.joblib
├── nysc_faq_dataset.xlsx
├── NYSC_FAQ_INTENT_CHATBOT_ipynb.txt
├── requirements.txt
└── README.md
```

### File descriptions

| File                                | Description                                       |
| ----------------------------------- | ------------------------------------------------- |
| `app.py`                            | Streamlit application and chatbot inference logic |
| `nysc_pipeline.joblib`              | Serialized trained TF-IDF + SVM pipeline          |
| `nysc_faq_dataset.xlsx`             | FAQ dataset used by the chatbot                   |
| `NYSC_FAQ_INTENT_CHATBOT_ipynb.txt` | Project development notebook exported as text     |
| `requirements.txt`                  | Python dependencies                               |
| `README.md`                         | Project documentation                             |

The project requirements specify `joblib`, `numpy`, `pandas`, `scikit-learn`, and `streamlit`.


# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/NYSC-FAQ-Intent-Chatbot.git

cd NYSC-FAQ-Intent-Chatbot
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Make sure the following files are located in the project directory:

```text
app.py
nysc_pipeline.joblib
nysc_faq_dataset.xlsx
requirements.txt
```

Then run:

```bash
streamlit run app.py
```

Streamlit will launch the chatbot application locally.


# 🧪 Example Queries

The chatbot was interactively tested with questions such as:

### Example 1 — General

```text
User:
What is NYSC?
```

```text
Detected intent:
general

Confidence:
0.99
```

### Example 2 — Allowance

```text
User:
When will the NYSC allowance be paid?
```

```text
Detected intent:
allowance

Confidence:
1.00
```

### Example 3 — Clearance

```text
User:
How long does clearance take?
```

```text
Detected intent:
clearance

Confidence:
0.99
```

### Example 4 — Unknown Query

```text
User:
Can I Japa?
```

```text
Detected intent:
fallback

Category:
Uncertain
```

These examples are from the notebook's interactive chatbot evaluation.

---

# 🚀 Deployment

The project was designed with deployment in mind.

The trained model is serialized into:

```text
nysc_pipeline.joblib
```

The notebook also demonstrates preparing the Streamlit application and identifies the required deployment files:

```text
app.py
nysc_pipeline.joblib
requirements.txt
```

These can be uploaded to a GitHub repository and used for deployment through a Streamlit hosting service.

---

# 🔐 Important Deployment Note

The current `app.py` references the dataset using:

```python
/content/nysc_faq_dataset.xlsx
```

This path is appropriate for the Google Colab environment used during development but should be changed to a repository-relative path before external deployment.

For example:

```python
df = pd.read_excel("nysc_faq_dataset.xlsx")
```

Similarly, ensure that:

```text
nysc_pipeline.joblib
```

is available in the application's working directory.

---

# 🧩 Key Technologies

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Support Vector Machine
* Logistic Regression
* Multinomial Naive Bayes

### Natural Language Processing

* TF-IDF Vectorization
* Text normalization
* Cosine similarity
* Intent classification

### Data Processing

* Pandas
* NumPy

### Model Persistence

* Joblib

### Application

* Streamlit

---

# 🌟 Key Features

* ✅ Natural-language FAQ interaction
* ✅ Intent classification
* ✅ TF-IDF text representation
* ✅ SVM-based classification
* ✅ FAQ similarity matching
* ✅ Confidence scoring
* ✅ Fallback handling for uncertain queries
* ✅ Duplicate FAQ removal
* ✅ Serialized model pipeline
* ✅ Streamlit web interface
* ✅ Interactive chatbot testing
* ✅ Confusion-matrix evaluation
* ✅ Deployment-ready model artifact

---

# 📚 What This Project Demonstrates

This project demonstrates practical application of an end-to-end machine-learning workflow:

```text
Problem Definition
       ↓
Dataset Preparation
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Model Training
       ↓
Model Comparison
       ↓
Hyperparameter Tuning
       ↓
Model Evaluation
       ↓
Pipeline Creation
       ↓
Model Serialization
       ↓
Chatbot Development
       ↓
Streamlit Deployment
```

It therefore goes beyond simply training a classification model and demonstrates how a machine-learning model can be converted into an interactive application.

---

# ⚠️ Limitations

Although the chatbot achieved strong test performance, it has several limitations.

### 1. FAQ-based knowledge

The chatbot can only provide answers contained within its FAQ dataset. It is not a general-purpose conversational AI system.

### 2. Dataset dependency

The quality of responses depends heavily on the quality, coverage, and accuracy of the underlying FAQ dataset.

### 3. Limited language understanding

The system relies primarily on TF-IDF representations and lexical similarity rather than transformer-based semantic embeddings.

### 4. Static knowledge

The model does not automatically retrieve current information from NYSC's official systems or external databases.

### 5. Ambiguous queries

Questions that are significantly different from the training examples may trigger the fallback mechanism or occasionally produce an incorrect classification.

---

# 🔮 Future Improvements

Potential future improvements include:

* [ ] Expand the FAQ dataset with more real-world questions.
* [ ] Add transformer-based embeddings such as Sentence-BERT.
* [ ] Introduce retrieval-augmented generation (RAG).
* [ ] Connect the chatbot to an updatable knowledge base.
* [ ] Add official NYSC information sources.
* [ ] Implement conversation history.
* [ ] Add multilingual support.
* [ ] Improve handling of Nigerian English and common Nigerian expressions.
* [ ] Add automated model monitoring.
* [ ] Create an API using FastAPI or Flask.
* [ ] Add unit and integration tests.
* [ ] Improve the Streamlit UI.
* [ ] Add analytics for unanswered questions.
* [ ] Periodically retrain the classifier with newly collected queries.

---

# 🎓 Project Context

This project was developed as an **AI/NLP machine-learning capstone project** demonstrating the development of an intent-classification chatbot from dataset preparation through model deployment.

The project covers the major components expected in a practical machine-learning application:

**Dataset → Preprocessing → NLP → Model Training → Evaluation → Model Serialization → Application → Deployment**

---

# 👨‍💻 Author

**Okotie Oritsematosan Jeremiah**

Physics Graduate | Data Science & AI/ML | AI Automation Engineering

### Areas of Interest

* Artificial Intelligence
* Machine Learning
* Natural Language Processing
* Data Science
* AI Automation
* Intelligent Applications

---

# 📄 License

This project is intended for educational and demonstration purposes.

If you choose to publish this repository under an open-source license, the **MIT License** is a suitable option.

---

# ⭐ Acknowledgements

* National Youth Service Corps (NYSC) — domain context
* Scikit-learn — machine-learning framework
* Pandas & NumPy — data processing
* Streamlit — application interface
* Joblib — model serialization
* Python community — supporting ecosystem

---

## ⭐ If You Find This Project Useful

If this project demonstrates something useful to you, consider giving the repository a **star ⭐** and sharing your feedback.

---

