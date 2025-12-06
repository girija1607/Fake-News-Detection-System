# Fake News Detection System (ML + NLP + Streamlit)

This project is an end-to-end Fake News Detection System built using Python, Machine Learning, and Natural Language Processing (NLP). The application allows users to paste any news headline or article and predicts whether the news is FAKE or REAL along with a confidence score. The trained model is deployed using a Streamlit web application.

Features:
- Predicts whether news is FAKE or REAL  
- Shows confidence score (probability)  
- Uses TF-IDF + Logistic Regression  
- Handles low-confidence predictions responsibly  
- Simple and clean Streamlit web UI  
- Fully built using free & open-source tools  

Machine Learning Approach:
Problem Type: Supervised Text Classification  
Input: News article or headline (text)  
Output: FAKE or REAL label  

Steps Followed:
Data Collection from Kaggle (Fake.csv + True.csv)  
Text Cleaning & NLP Preprocessing  
Feature Extraction using TF-IDF  
Model Training using Logistic Regression  
Model Evaluation (Accuracy, Precision, Recall, F1-score)  
Model Deployment using Streamlit  

Project Folder Structure:
fake-news-detector/
├── training.py
├── app.py
├── fake_news_model.pkl
└── data/
    ├── Fake.csv
    └── True.csv

Tech Stack Used:
Programming Language: Python  
Machine Learning: Scikit-learn  
NLP: NLTK, TF-IDF  
Data Handling: Pandas, NumPy  
Model Saving: Joblib  
Web App: Streamlit  
Dataset Source: Kaggle  
Deployment: Streamlit Cloud (optional)  

All tools used in this project are 100% free and open-source.

NLP Preprocessing Steps:
Lowercasing text  
Removing URLs  
Removing special characters & numbers  
Stopwords removal using NLTK  
Text vectorization using TF-IDF  
Used Unigrams & Bigrams (1,2-grams)  

Model Used: Logistic Regression
Fast and efficient for text classification  
Works very well with TF-IDF features  
Handles large datasets easily  
Outputs probability scores  
Easy to interpret  

Model Performance:
Accuracy: ~99%  
High Precision & Recall  
Strong Confusion Matrix results  

Confusion Matrix:
Actual FAKE → Predicted FAKE: 4641  
Actual FAKE → Predicted REAL: 54  
Actual REAL → Predicted FAKE: 37  
Actual REAL → Predicted REAL: 4246  

Confidence Threshold:
A confidence threshold of 60% is applied.  
If confidence < 60% → Model shows "Not sure"  
If confidence ≥ 60% → Shows FAKE or REAL clearly  

How to Run the Project Locally:

Step 1: Clone the Repository
git clone https://github.com/girija1607/Fake-News-Detection-System.git

cd fake-news-detector

Step 2: Create Virtual Environment
python -m venv venv
venv\Scripts\activate

Step 3: Install Required Libraries
pip install pandas numpy scikit-learn nltk streamlit joblib

Step 4: Train the Model
python training.py

Step 5: Run the Web App
streamlit run app.py

Open in browser:
http://localhost:8501



Key Concepts Used:
Supervised Machine Learning  
Text Classification  
Natural Language Processing (NLP)  
TF-IDF (Term Frequency – Inverse Document Frequency)  
Logistic Regression  
Train-Test Split  
Accuracy, Precision, Recall, F1-score  
Confusion Matrix  
Probability & Confidence Score  
Model Deployment using Streamlit  

Disclaimer:
This project is a machine learning demo trained on a public dataset. Predictions may not always be correct and should not be used as the only source of truth for verifying news.

Limitations:
Trained mainly on US political news  
Limited exposure to Indian news  
Works only in English  
Cannot detect sarcasm or memes  
Model may misclassify unseen writing styles  

Future Improvements:
Train with Indian news datasets  
Add Multilingual support (Hindi + English)  
Use Deep Learning (LSTM, BERT)  
Add LLM-based explanation  
Build a Browser Extension  

Author:
Girija Singhal  
Artificial Intelligence & Full Stack Developer  
Email: girijasinghal1607@gmail.com  
GitHub: (https://github.com/girija1607)  


