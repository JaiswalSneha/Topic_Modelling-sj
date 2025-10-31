import pandas as pd
import numpy as np
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
from collections import Counter
from pathlib import Path
import streamlit as st
st.set_page_config(layout="wide")
import sys
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import joblib

#=======================================================================
sys.path.append(os.path.dirname(__file__))
data_dir = Path(__file__).parent / 'data'

# df dataset
df = data_dir / 'train_clean.csv'
df = pd.read_csv(df)

# ====================== Prediction Section ============================

st.subheader("🔍 Predict Topic for a New Document")
st.markdown("---")

# Load the saved model and vectorizer

sys.path.append(os.path.dirname(__file__))
cache_dir = Path(__file__).parent / 'cache'
lda_model = cache_dir / 'lda_model.joblib'
vectorizer = cache_dir / 'vectorizer.joblib'
list_topics = cache_dir / 'list_topics.joblib'

lda = joblib.load(lda_model)
vectorizer = joblib.load(vectorizer)
list_topics = joblib.load(list_topics)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)


#=================================== Word cloud ==============================================
def display_wordcloud(model, feature_names, num_top_words_100):
    st.subheader('WordCloud Generation')

    columns = st.columns(5)

    for topic_idx, topic in enumerate(model.components_):
        col = columns[topic_idx % 5]
        with col:
            st.markdown(f"**Topic {topic_idx}**")

            top_features_ind = topic.argsort()[:-num_top_words_100 - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]

            word_freq = {feature_names[i]: topic[i] for i in top_features_ind}

            wordcloud = WordCloud(
                width=400,
                height=300,
                background_color='white',
                colormap='viridis'
            ).generate_from_frequencies(word_freq)

            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

    
num_top_words_100 = 50
feature_names = vectorizer.get_feature_names_out()
display_wordcloud(lda, feature_names, num_top_words_100)
st.markdown('---')
#=============================================================================================

# --- User input ---
new_doc = st.text_input("Enter a document to classify:")
#===================================================================

col1, col2, col3, col4, col5 = st.columns(5)
topic_list = {}
with col1:
    topic_list[0] = st.selectbox('Topic 0',df['Topic'].unique())
with col2:
    topic_list[1] = st.selectbox('Topic 1',df['Topic'].unique())
with col3:
    topic_list[2] = st.selectbox('Topic 2',df['Topic'].unique())
with col4:
    topic_list[3] = st.selectbox('Topic 3',df['Topic'].unique())
with col5   :
    topic_list[4] = st.selectbox('Topic 4',df['Topic'].unique())


predict_button = st.button('Predict')
if predict_button:
    st.markdown('---')
    if new_doc:
        processed_doc = preprocess(new_doc)
        new_doc_vectorized = vectorizer.transform([processed_doc])
        topic_distribution = lda.transform(new_doc_vectorized)
        dominant_topic = np.argmax(topic_distribution)

        st.markdown(f"**Most likely topic:** {topic_list[dominant_topic]}")
        st.write("**Topic distribution:**", topic_distribution)


st.markdown('---')