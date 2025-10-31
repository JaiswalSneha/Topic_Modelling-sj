import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
from pathlib import Path
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import streamlit as st
import sys
import time
import matplotlib.pyplot as plt
import re

start = time.time()

# ====================== Streamlit Setup ======================
st.set_page_config(layout="wide")
st.title("Data Preprocessing")
st.markdown('---')

#================================INTRO ======================================

st.markdown("""
### About the Dataset

This project uses the **BBC News Dataset**, a well-known collection of news articles provided for **machine learning and natural language processing research**.

The dataset contains:
- **News articles** originally published by the **BBC**
- **Categories** such as *business, entertainment, politics, sport,* and *tech*
- Each record includes a **title**, **text content**, and its **category label**

""")
st.markdown('---')

# ====================== Importing the datasets ===========================
sys.path.append(os.path.dirname(__file__))
data_dir = Path(__file__).parent / 'data'

# df dataset
df = data_dir / 'bbc_news.csv'
df = pd.read_csv(df)

st.markdown('**Shape of the dataset**')
st.text(df.shape)

st.markdown('**Overview of the dataset**')
st.dataframe(df)

df.rename(columns={'Description': 'desc'}, inplace=True)
st.markdown("*Renaming the columns for ease of use ={'Description': 'desc'}*")

st.markdown('---')
st.subheader('Duplicates')
st.text(f"Based on Title : {df.duplicated(subset='Title').sum()}")
st.text(f"Based on Description : {df.duplicated(subset='desc').sum()}")
st.markdown('---')

st.text('Title Based Duplicates')
st.dataframe(df[df.duplicated(subset='Title', keep=False)].sort_values('Title'))
st.text('Description Based Duplicates')
st.dataframe(df[df.duplicated(subset='desc', keep=False)].sort_values('desc'))

st.markdown('---')
st.subheader("After Removing the duplicates")
df = df.drop_duplicates(subset='desc')
df = df.drop_duplicates(subset='Title')

st.markdown('**After Shape of the dataset**')
st.text(df.shape)
st.text('Duplicates')
st.text(f"Based on Title : {df.duplicated(subset='Title').sum()}")
st.text(f"Based on Description : {df.duplicated(subset='desc').sum()}")
st.markdown('---')

# =============================== Data Preprocessing ====================================

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return tokens


df['org_desc'] = df['desc']
df['desc'] = df["Title"] + " " + df["desc"]

df['clean'] = df['desc'].apply(preprocess)

st.subheader("Text Preprocessing Steps Applied")

st.markdown("""
1. **Lowercasing:** All text is converted to lowercase to treat words like 'Data' and 'data' the same.
2. **Tokenization:** Text is split into individual words (tokens).
3. **Stopword Removal:** Common English words such as 'the', 'is', 'and' that carry little meaning are removed.
4. **Lemmatization:** Words are reduced to their base form, e.g., 'running' → 'run', 'cars' → 'car'.
5. **Alphabetic Filtering:** Only words made of letters are kept; numbers, punctuation, and symbols are removed.

**Result:** Each document is cleaned and tokenized, stored in the `clean` column, ready for Topic modeling.
""")

st.dataframe(df)

sys.path.append(os.path.dirname(__file__))
data_dir = Path(__file__).parent / 'data'
# loc = data_dir / 'train_clean.csv'

# df.to_csv(loc)


#============================ EDA =======================================

st.markdown('---')
st.header('EDA')

st.write("Total documents:", len(df))
st.write("Average document length:", df['desc'].str.split().apply(len).mean())

st.markdown('---')

st.subheader('Contribution of each topic')

col1,col2 = st.columns(2)
with col1:
    fig,ax = plt.subplots()
    plt.bar(df['Topic'].value_counts().index,df['Topic'].value_counts().values)
    st.pyplot(fig)
with col2:
    st.dataframe(pd.DataFrame({'Topic':df['Topic'].value_counts().index,'Topic count':df['Topic'].value_counts().values}))

st.markdown("""
The dataset contains text documents categorized into **five major topics**: Business, Sport, Politics, Entertainment and Tech

---

### 🧠 Insights:
- The dataset is **fairly balanced**, with each topic having a comparable number of documents.
- **Business** and **Sport** dominate slightly, suggesting these themes might have a stronger influence during topic modeling.
              
""")

st.markdown('---')


st.header("Most Common Unigrams, Bigrams, and Trigrams")
from collections import Counter
all_words = [w for doc in df['clean'] for w in doc]
common_words = Counter(all_words).most_common(50)
df_common = pd.DataFrame(common_words, columns=["word", "count"]).sort_values(by="count", ascending=False)
st.bar_chart(df_common.set_index("word"))

#=============================n grams============================
from nltk import ngrams

tokens = [w for doc in df['clean'] for w in doc]

def plot_top_ngrams(tokens, n, title):
    n_grams = ngrams(tokens, n)
    common_ngrams = Counter([' '.join(gram) for gram in n_grams]).most_common(25)
    df_common = pd.DataFrame(common_ngrams, columns=["ngram", "count"]).sort_values(by="count", ascending=False)
    st.subheader(f" Top {title}")
    st.bar_chart(df_common.set_index("ngram"))

col1, col2 = st.columns(2)
with col1:
    plot_top_ngrams(tokens, 2, "Bigrams (Two-Word Phrases)")
with col2:
    plot_top_ngrams(tokens, 3, "Trigrams (Three-Word Phrases)")


#========================== word cloud : overall ============================

st.markdown('---')
st.header("Wordcloud : For all the documents")
from wordcloud import WordCloud

def preprocess(text):
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    tokens = [w for w in tokens]
    return tokens

df['tokens'] = df['org_desc'].apply(preprocess)

col1,col2 = st.columns(2)
with col1:
    st.text('Before Stopwords')
    text = " ".join([" ".join(words) for words in df['tokens']])
    st.image(WordCloud(width=800, height=400, background_color='white').generate(text).to_array())
with col2:
    st.text('After Stopwords')
    text = " ".join([" ".join(words) for words in df['clean']])
    st.image(WordCloud(width=800, height=400, background_color='white').generate(text).to_array())


st.markdown('---')

#========================== word cloud : topic wise ============================

st.header('WordCloud : Topic wise')
st.text('After Stop words removal')

topics = df['Topic'].unique()

# Loop through topics in pairs (2 per row)
for i in range(0, len(topics), 2):
    col1, col2 = st.columns(2)
    
    # --- First column ---
    with col1:
        topic = topics[i]
        st.markdown(f"### 🗂️ {topic}")
        topic_words = ' '.join([' '.join(words) for words in df[df['Topic'] == topic]['clean']])
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(topic_words)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    
    # --- Second column (only if another topic exists) ---
    if i + 1 < len(topics):
        with col2:
            topic = topics[i + 1]
            st.markdown(f"### 🗂️ {topic}")
            topic_words = ' '.join([' '.join(words) for words in df[df['Topic'] == topic]['clean']])
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(topic_words)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)




#========================== checking the word "said"===========================

st.markdown('---')
st.subheader('Enter word to check occurences in the topics')
topic_col = 'clean'
word_to_check = st.text_input("Enter the word to check", value="said")

# ==================== Processing ====================

if word_to_check:
    # Count occurrences of the word per row
    df["word_count"] = df["clean"].apply(lambda x: x.count(word_to_check))
    
    # Aggregate counts by topic
    word_summary = df.groupby("Topic")["word_count"].sum().reset_index()
    word_summary.rename(columns={"word_count": f"Total Occurrences of '{word_to_check}'"}, inplace=True)
    
    # Display
    st.table(word_summary.sort_values(word_summary.columns[1], ascending = False))


st.markdown("*Total time taken:*")
end = time.time()
st.text(end-start)