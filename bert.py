import streamlit as st
import pandas as pd
import time
from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import plotly.express as px

# ====================== Streamlit Setup ======================
st.set_page_config(layout="wide")
st.title("BERTopic - BERT-based Topic Modeling")
st.markdown('---')

# ====================== Load Dataset ======================
data_dir = Path(__file__).parent / 'data'
df_path = data_dir / 'train_clean.csv'

if not df_path.exists():
    st.error("'train_clean.csv' not found in the 'data' folder.")
    st.stop()

df = pd.read_csv(df_path)

# Expecting a text column (change if different)
if 'clean' not in df.columns:
    st.error("The dataset must contain a column named 'clean'.")
    st.stop()

st.write(f"Loaded **{len(df)}** documents.")

# ====================== User Controls ======================
st.header("Model Settings")

embedding_model = st.selectbox(
    "Select embedding model",
    ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L12-v2", "all-mpnet-base-v2"]
)

min_topic_size = st.slider("Minimum topic size", 5, 100, 10)
n_gram_range = st.selectbox("N-gram range", [(1, 1), (1, 2)], index=0)

# ====================== Train BERTopic ======================
if st.button("Train BERTopic Model"):
    start = time.time()

    with st.spinner("Training BERTopic model...⏳"):
        docs = df['clean'].astype(str).tolist()

        # Initialize BERTopic
        topic_model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=min_topic_size,
            n_gram_range=n_gram_range,
            verbose=True
        )

        topics, probs = topic_model.fit_transform(docs)

    st.success(f"Model trained in {round(time.time()-start, 2)} seconds")

    # Save model
    import os
    import sys
    sys.path.append(os.path.dirname(__file__))
    cache_dir = Path(__file__).parent / 'cache'
    bertopic_model_loc = cache_dir / 'bertopic_model'
    topic_model.save(str(bertopic_model_loc))
    st.info("Model saved as 'bertopic_model'")

    # ====================== Display Topics ======================
    st.subheader("Topic Overview")
    topic_info = topic_model.get_topic_info()
    st.dataframe(topic_info)

    # ====================== Top Words per Topic ======================
    st.subheader("Top Words in Each Topic")

    for topic_id in topic_info.head(10)['Topic']:  # Show first 10 topics
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        st.markdown(f"**Topic {topic_id}:** {', '.join([w[0] for w in words])}")

    # ====================== Visualization ======================
    st.markdown('---')
    st.subheader("Topic Visualization")

    fig = topic_model.visualize_topics()
    st.plotly_chart(fig, use_container_width=True)

# ====================== Prediction Section ======================
st.markdown('---')
st.header("Predict Topic for a New Document")

new_doc = st.text_area("Enter your document text:")

if st.button("Predict Topic"):
    try:
        topic_model = BERTopic.load(str(bertopic_model_loc))
    except Exception as e:
        st.error("Model not found. Please train the BERTopic model first.")
        st.stop()

    if new_doc.strip():
        new_topic, new_prob = topic_model.transform([new_doc])
        st.write(f"**Predicted Topic:** {new_topic[0]}")
        st.write(f"**Confidence:** {round(float(new_prob[0]), 3)}")

        topic_words = topic_model.get_topic(new_topic[0])
        if topic_words:
            st.markdown("**Top Words in Predicted Topic:**")
            st.write(", ".join([w[0] for w in topic_words]))
    else:
        st.warning("Please enter some text for prediction.")
