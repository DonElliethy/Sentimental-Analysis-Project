# Sentimental Analysis Project
Sophomore Second Semester AI Course Project

This repository contains a comprehensive pipeline for analyzing social media sentiment using various Machine Learning and Natural Language Processing (NLP) techniques. The project focuses on classifying diverse emotional labels into primary sentiment categories: Positive, Negative, and Neutral.

📊 Dataset Overview
The project utilizes a custom dataset (sentimentdataset.csv) containing 732 social media entries. Each entry includes:

Text: The actual post content.

Sentiment (Label): High-granularity labels (e.g., Euphoria, Melancholy, Zest, Bitter).


Metadata: Timestamp, User, Source (Twitter, Instagram, Facebook), Topic, Retweets, Likes, and Country. 

🛠️ Project Workflow
1. Preprocessing & Text Cleaning
The notebook implements a robust text-cleaning function that handles:


Tokenization: Breaking text into individual words. 


Noise Removal: Eliminating punctuation and unorganized strings. 


Emoji Handling: Converting emojis to descriptive text using demojize. 


Stopword Removal: Filtering out common English words (using NLTK). 


Lemmatization: Reducing words to their base forms (roots) using WordNetLemmatizer with POS tagging. 

2. Sentiment Mapping (Label Consolidation)
The original dataset contains over 200 unique emotional labels. The project simplifies these into a three-class system:


Automated Classification: Uses SentimentIntensityAnalyzer (VADER) to assign a compound score to labels. 


Logic-Based Refinement: Custom logic handles edge cases (e.g., mapping Nostalgia or Awe to Positive, and Overwhelmed or Heartache to Negative) that automated tools might misclassify as Neutral. 

3. Feature Engineering

TfidfVectorizer: Converts cleaned text into numerical vectors based on word importance. 

Data Distribution: After cleaning, the dataset resulted in:

Positive: 496 entries

Negative: 195 entries


Neutral: 41 entries 

🚀 Models Implemented
The project explores multiple classification algorithms:


Multinomial Naive Bayes: Ideal for discrete text features. 


Logistic Regression: Baseline linear classification. 

Random Forest Classifier: Used for robust ensemble learning.


Support Vector Machines (SVM): For high-dimensional feature spaces. 


K-Nearest Neighbors (KNN): Distance-based classification. 


Deep Learning (LSTM): A sequential model using tensorflow.keras with Embedding and LSTM layers for capturing context. 

📈 Evaluation
The models are evaluated using:

Accuracy Score: Overall prediction correctness.


Precision, Recall, & F1-Score: To account for the class imbalance. 


Confusion Matrix: Visualization of true vs. predicted labels. 


Cross-Validation: Ensuring model stability across different data splits. 

💻 Requirements
To run this notebook, you will need:

pandas, numpy

matplotlib, seaborn

scikit-learn

nltk (Stopwords, VADER, WordNet)

emoji

tensorflow (for the LSTM model)

📁 Files
Project NoteBook.ipynb: The main Jupyter notebook containing the code.

sentimentdataset.csv: The raw dataset used for training and testing.


trained_model.joblib: The saved Random Forest model for future inference. 


vectorizer.joblib: The saved TF-IDF vectorizer.
