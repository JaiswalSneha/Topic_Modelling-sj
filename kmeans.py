import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
import sys
from pathlib import Path
import os
import time

start = time.time()

# ====================== Streamlit Setup ======================
st.set_page_config(layout="wide")
st.title("KMeans using TFIDF & BOW")
st.markdown('---')

# ====================== Importing the datasets ===========================
sys.path.append(os.path.dirname(__file__))
data_dir = Path(__file__).parent / 'data'

# df dataset
df = data_dir / 'train_clean.csv'
df = pd.read_csv(df)

# ====================== Parameter Selection ===========================

st.header("K-Means Clustering")
st.subheader("Feature Extraction")
model = st.selectbox('Select the vectorizer',['BOW','TFIDF'])

col1, col2 = st.columns(2)
with col1:
    inp_ngrams = st.selectbox("Select the Ngrams",[(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)],index=0)
    inp_min_df = st.number_input("Minimum document frequency (min_df)", min_value=1, value=5, step=1)
    inp_max_df = st.number_input( "Maximum document frequency (max_df, proportion in %)", min_value=5, max_value=100, value=70, step=5 )
    inp_max_df = inp_max_df / 100
with col2:
    inp_max_features = st.number_input("Max features (vocabulary size)", min_value=1000, value=15000, step=1000)
    inp_token_len = st.number_input("Minimum token length", min_value=3, value=3, step=1)
    num_top_words = st.number_input('Select the max number of top words to be displayed per topic',min_value=1, max_value=500,value=10)

num_clusters = st.slider("Select number of topics (clusters)", 2, 10, 5)

import ast
df['get_word'] = df['clean'].apply(ast.literal_eval)

unq_words = [w for doc in df['get_word'] for w in doc]
unq_words = set(unq_words)
exclude_words = st.multiselect("Select words to exclude from the corpus (optional) :",options=unq_words)

def remove_words_from_strlist(text, words_to_remove):
    try:
        tokens = ast.literal_eval(text)
        tokens = [w for w in tokens if w not in words_to_remove]
        return str(tokens)
    except:
        return text
df['filtered'] = df['clean'].apply(lambda x: remove_words_from_strlist(x, exclude_words))


corpus = df['filtered'].tolist()

ngram_button = st.button('Proceed')

# ==================================== button on click ================================

if ngram_button:

    start = time.time()
    vectorizer = TfidfVectorizer(ngram_range=(inp_ngrams[0],inp_ngrams[1]), stop_words='english',min_df=inp_min_df,max_df=inp_max_df,max_features=inp_max_features,token_pattern=r'(?u)\b[a-zA-Z]{3,}\b')
    X = vectorizer.fit_transform(corpus)
    vocab_size = len(vectorizer.vocabulary_)
    st.markdown(f"**Vocabulary size:** {vocab_size}")
    end = time.time()
    st.markdown(f"**Time taken to create the Corpus and Vocabulary** {end - start}")

    st.markdown(f"Ignored rare words **(<{inp_min_df} docs)** and very common words **(>{inp_max_df*100}% docs)**")
    st.markdown(f"Only included alphabetic tokens of length **{inp_token_len}+**")

    start= time.time()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    end = time.time()
    st.markdown(f"**Time taken to train the model** {end - start}")

    # =============================== Top words per cluster ===============================
    st.markdown("---")
    st.subheader("Top Words in Each Topic")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    for i in range(num_clusters):
        top_words = [terms[ind] for ind in order_centroids[i, :10]]
        st.markdown(f"**Topic {i+1}:** {', '.join(top_words)}")

    # =============================== WordCloud Visualization =====================================
    st.markdown("---")
    st.subheader("WordClouds per Topic")
    cols = st.columns(2)
    for i in range(num_clusters):
        top_words = {terms[ind]: kmeans.cluster_centers_[i, ind] for ind in order_centroids[i, :50]}
        wordcloud = WordCloud(width=600, height=300, background_color='white', colormap='plasma').generate_from_frequencies(top_words)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        with cols[i % 2]:
            st.markdown(f"**Topic {i+1}**")
            st.pyplot(fig)

    # --- Assign Topics to Documents ---
    df["kmeans_topic"] = labels
    st.markdown("---")

    #=============================== Distribution of topics visualization ==========================

    st.subheader("Distribution of True Topics vs KMeans Clusters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### KMeans Topic Distribution")
        fig, ax = plt.subplots(figsize=(8,5))
        kmeans_counts = df["kmeans_topic"].value_counts().sort_index()
        ax.bar(kmeans_counts.index.astype(str), kmeans_counts.values, color='blue')
        ax.set_xlabel("KMeans Topics")
        ax.set_ylabel("Document Count")
        st.pyplot(fig)

    with col2:
        st.markdown("##### True Topic Distribution")
        fig, ax = plt.subplots(figsize=(8,5))
        topic_counts = df["Topic"].value_counts().sort_index()
        ax.bar(topic_counts.index.astype(str), topic_counts.values, color='green')
        ax.set_xlabel("True Topics")
        ax.set_ylabel("Document Count")
        st.pyplot(fig)