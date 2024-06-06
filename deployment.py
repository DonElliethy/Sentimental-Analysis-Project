import joblib
import streamlit as st
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import string
import emoji
import nltk

# Loading model
clf = joblib.load('D:/FCIS/Sophomore/Second Semester/AI/AI_Project/trained_model.joblib')

def preprocess(text):
    tokens = word_tokenize(text)
    stopwords_list = set(stopwords.words('english'))
    punctuation = string.punctuation
    text = ' '.join(tokens)  # Join tokens into a single string
    text = text.translate(str.maketrans("", "", punctuation))  # Remove punctuation
    text = emoji.demojize(text)  # Remove emojis
    tokens = [token.lower() for token in text.split() if token.lower() not in stopwords_list]
    return tokens

lemmatizer = WordNetLemmatizer()
from nltk.corpus import wordnet
def get_wordnet_pos(tag):
    tag = tag[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
def Lemma(x):

    lemmatizer = WordNetLemmatizer()
    lemmatized_entry = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) for token, tag in nltk.pos_tag(x)]
    return lemmatized_entry
    

def removeBrackets(x):
    return [' '.join(str(word) for word in x).replace('[','').replace(']','').replace("'", '').replace(',', '')]

def tfidf(x):
    vectorizer = joblib.load('D:/FCIS/Sophomore/Second Semester/AI/AI_Project/vectorizer.joblib')
    tfidf_vectors = vectorizer.transform(x)
    tfidf_df = pd.DataFrame(tfidf_vectors.toarray(), columns=vectorizer.get_feature_names_out())
    return tfidf_df

def transform(x):
    return tfidf(removeBrackets(Lemma(preprocess(x))))

def classify(x) -> str:
    if x == 1:
        return "positive"
    elif x == 0:
        return "neutral"
    else:    
        return "negative"

# GUI
st.title("Sentiment Analysis")

st.subheader("Please enter your text below")
txt = st.text_area(" ")
clicked = st.button("Predict")

if clicked:
    result = clf.predict(transform(txt))
    st.write(classify(result))
