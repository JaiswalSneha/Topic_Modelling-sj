import pandas as pd
import numpy as np
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
from collections import Counter
from pathlib import Path
import streamlit as st
import sys
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary


start_start = time.time()

# ====================== Streamlit Setup ======================
st.set_page_config(layout="wide")
st.title("LDA using BOW")
st.markdown('---')

# ====================== Importing the datasets ===========================
sys.path.append(os.path.dirname(__file__))
data_dir = Path(__file__).parent / 'data'

# df dataset
df = data_dir / 'train_clean.csv'
df = pd.read_csv(df)

# ====================== User Selects Ngrams ===========================



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

import ast
df['get_word'] = df['clean'].apply(ast.literal_eval)

unq_words = [w for doc in df['get_word'] for w in doc]
unq_words = set(unq_words)
exclude_words = st.multiselect("Select words to exclude from the corpus:",options=unq_words)

def remove_words_from_strlist(text, words_to_remove):
    try:
        tokens = ast.literal_eval(text)
        tokens = [w for w in tokens if w not in words_to_remove]
        return str(tokens)
    except:
        return text
df['filtered'] = df['clean'].apply(lambda x: remove_words_from_strlist(x, exclude_words))


# change = pd.DataFrame({'values': list(unq_words)})
# change.to_csv('my_set.csv', index=False)


corpus = df['filtered'].tolist()

ngram_button = st.button('Proceed')

# ====================== BOW ===========================

num_topics = 5

cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)

if ngram_button:
    st.session_state.button_clicked = True
    
    start= time.time()
    vectorizer = CountVectorizer(ngram_range=(inp_ngrams[0],inp_ngrams[1]), stop_words='english',min_df=inp_min_df,max_df=inp_max_df,max_features=inp_max_features,token_pattern=r'(?u)\b[a-zA-Z]{3,}\b')
    X = vectorizer.fit_transform(corpus)
    vocab_size = len(vectorizer.vocabulary_)
    st.markdown(f"**Vocabulary size:** {vocab_size}")
    end = time.time()
    st.markdown(f"**Time taken to create the Corpus and Vocabulary** {end - start}")

    start= time.time()
    lda = LatentDirichletAllocation(n_components=num_topics,random_state=42,learning_method='online')
    lda.fit(X)
    end = time.time()
    st.markdown(f"**Time taken to train the model** {end - start}")
    st.markdown(f"Ignored rare words **(<{inp_min_df} docs)** and very common words **(>{inp_max_df*100}% docs)**")
    st.markdown(f"Only included alphabetic tokens of length **{inp_token_len}+**")

    # ========================== Data Visualization ===========================

    start = time.time()
    
    list_topics = []

    st.markdown('---')

    def display_topics(model, feature_names,num_top_words):
        for topic_idx, topic in enumerate(model.components_):
                # --- Get top features and their weights ---
                top_features_ind_user = topic.argsort()[:-num_top_words - 1:-1]
                top_features_user = [feature_names[i] for i in top_features_ind_user]
                top_weights_user = topic[top_features_ind_user]

                # --- Display top words as text ---
                var = [", ".join(top_features_user)]
                list_topics.append(var) 


    feature_names = vectorizer.get_feature_names_out()
    display_topics(lda, feature_names,num_top_words)


    st.subheader(f"Displaying the top {num_top_words} words in each topic")

    col1, col2 = st.columns(2)
    for i, topic_words in enumerate(list_topics[:5]):  # Limit to 5 topics
        output = ", ".join(map(str, topic_words))
        if i % 2 == 0:
            with col1:
                st.markdown(f"**Topic {i+1}**")
                st.text(output)
        else:
            with col2:
                st.markdown(f"**Topic {i+1}**")
                st.text(output)


    st.markdown('---')

    def display_wordcloud(model, feature_names, num_top_words_100):
        st.subheader('WordCloud Generation')

        # Create two columns
        col1, col2 = st.columns(2)

        for topic_idx, topic in enumerate(model.components_):
            # Decide which column to use
            column = col1 if topic_idx % 2 == 0 else col2

            with column:
                st.subheader(f"Topic {topic_idx + 1}")

                # --- Get top features and their weights ---
                top_features_ind = topic.argsort()[:-num_top_words_100 - 1:-1]
                top_features = [feature_names[i] for i in top_features_ind]
                top_weights = topic[top_features_ind]

                # --- Create a dictionary for word cloud ---
                word_freq = {feature_names[i]: topic[i] for i in top_features_ind}

                # --- Generate the word cloud ---
                wordcloud = WordCloud(
                    width=600,
                    height=300,
                    background_color='white',
                    colormap='viridis'
                ).generate_from_frequencies(word_freq)

                # --- Display the word cloud ---
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)   
    
    num_top_words_100 = 100
    feature_names = vectorizer.get_feature_names_out()
    display_wordcloud(lda, feature_names, num_top_words_100)

    st.markdown('---')
#-------------------------------------------------------------------------------------

    import pyLDAvis
    import joblib
    from pathlib import Path

    st.title("LDA Visualization (pyLDAvis)")

    # Prepare data for visualization
    topic_term_dists = lda.components_
    doc_topic_dists = lda.transform(X)
    vocab = vectorizer.get_feature_names_out()
    doc_lengths = np.array(X.sum(axis=1)).flatten()
    term_frequency = np.array(X.sum(axis=0)).flatten()

    # Create the visualization
    lda_vis = pyLDAvis.prepare(
        topic_term_dists,
        doc_topic_dists,
        doc_lengths,
        vocab,
        term_frequency
    )

    cache_dir = Path(__file__).parent / 'cache'
    lda_html = cache_dir / 'lda.html'
    # Save to HTML and display in Streamlit
    pyLDAvis.save_html(lda_vis, str(lda_html))
    with open(str(lda_html), "r", encoding="utf-8") as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=800)

    st.markdown('---')

#==================================== coherencemodel ========================================================

    from gensim.models.coherencemodel import CoherenceModel
    import numpy as np

    st.markdown("""
    ### What is Topic Coherence?

    **Topic Coherence** measures how **meaningful and consistent** the words within a topic are — basically, it checks whether the top words in a topic make *sense together*.

    """)

    start = time.time()
    
    texts = [doc.split() for doc in df['filtered'].tolist()]
    dictionary = Dictionary(texts)
    corpus_gensim = [dictionary.doc2bow(t) for t in texts]

    class SklearnLDAWrapper:
        def __init__(self, lda_model):
            self.lda_model = lda_model
            
        def get_topics(self):
            return self.lda_model.components_ / self.lda_model.components_.sum(axis=1)[:, np.newaxis]

    # Wrap your sklearn LDA
    lda_wrapper = SklearnLDAWrapper(lda_model=lda)

    # Then use CoherenceModel
    coherence_model = CoherenceModel(model=lda_wrapper, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    

# ====================== printing information ============================

    st.subheader("📊 Coherence Score Interpretation")

    # Data for display
    data = {
        "Coherence Range": ["0.0–0.4", "0.4–0.6", "0.6–0.8", "0.8+"],
        "Interpretation": [
            "❌ Poor — topics are messy or unrelated",
            "⚠️ Fair — some meaningful patterns",
            "✅ Good — topics make sense to a human",
            "🌟 Excellent — very clear and coherent topics"
        ]
    }

    df_coherence = pd.DataFrame(data)
    st.dataframe(df_coherence.style)

    st.markdown(f"**Coherence Score as per selected parameters:** {coherence_score}")

    st.markdown("---")

#=============================================================
    st.text(f"Total time taken: {time.time()-start_start}")


#===================================================================
    import joblib

    sys.path.append(os.path.dirname(__file__))
    data_dir = Path(__file__).parent / 'cache'
    lda_model = data_dir / 'lda_model.joblib'
    vectorizer_path = data_dir / 'vectorizer.joblib'
    list_topics_path = data_dir / 'list_topics.joblib'

    # Save LDA model and vectorizer
    joblib.dump(lda, lda_model)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(list_topics, list_topics_path)
    
