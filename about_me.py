import streamlit as st
st.set_page_config(layout="wide")

# Basic page styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: white;
            color: #116A91;
        }
        .title {
            text-align: center;
            color: #116A91;
            font-size: 36px;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #444444;
            font-size: 18px;
            margin-bottom: 30px;
        }
        .section {
            background-color: #f9f9f9;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 6px;
        }
        .header {
            color: #116A91;
            font-size: 20px;
            margin-bottom: 10px;
        }
        .content {
            color: #333333;
            font-size: 16px;
            line-height: 1.6;
        }
    </style>
    """, unsafe_allow_html=True
)

# Page Title and Subtitle
st.markdown("<div class='title'>BBC News Topic Modeling App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Discover hidden themes within BBC News articles using advanced topic modeling techniques 🧠</div>", unsafe_allow_html=True)

# Creator Section (unchanged)
st.markdown("""
<div class='section'>
    <div class='header'>👤 About the Creator</div>
    <div class='content'>
        <strong>SJ</strong> is the creator of this Topic Modeling Application.
        <br><br>
        Currently working as a <strong>Product Manager</strong>, SJ specializes in leading cross-functional teams to build and launch user-centric digital products. With a strong foundation in both business strategy and data, SJ is passionate about solving real-world problems through thoughtful product design.
        <br><br>
        Prior to becoming a Product Manager, SJ held roles as:
        <ul>
            <li><strong>Data Analyst</strong> – Experienced in SQL, Excel, Tableau, Python, Consumer Bureau data, and Microfinance analytics</li>
            <li><strong>Data Engineer</strong> – Worked with IBM DataStage, Informatica, Ataccama, and Collibra</li>
        </ul>
        This diverse background is what inspired SJ to build tools like this — combining the power of data with simple, intuitive products.
    </div>
</div>
""", unsafe_allow_html=True)

# Purpose Section
st.markdown("""
<div class='section'>
    <div class='header'>📌 Purpose of This App</div>
    <div class='content'>
        The goal of this application is to explore and understand the **hidden topics** present in BBC News articles.  
        <br><br>
        It applies multiple topic modeling techniques to uncover semantic patterns and organize articles into meaningful groups:
        <ul>
            <li>🧮 <strong>LDA with Bag-of-Words (BOW)</strong> – Traditional topic modeling using word counts</li>
            <li>📊 <strong>LDA with TF-IDF</strong> – Focuses on important, distinguishing words across articles</li>
            <li>🔵 <strong>KMeans Topic Modeling</strong> – Clusters documents based on feature similarity</li>
            <li>🤖 <strong>BERT-based Topic Modeling</strong> – Uses transformer embeddings for context-aware clustering</li>
        </ul>
        This enables users to visualize, analyze, and interpret the key themes in BBC’s diverse news coverage.
    </div>
</div>
""", unsafe_allow_html=True)

# Dataset Section
st.markdown("""
<div class='section'>
    <div class='header'>🗞️ About the Dataset</div>
    <div class='content'>
        The dataset used in this app originates from <strong>BBC News</strong>, and is widely used as a benchmark for machine learning and NLP research.
        <br><br>
        It contains:
        <ul>
            <li>📰 News articles categorized into <em>business, entertainment, politics, sport,</em> and <em>tech</em></li>
            <li>📄 Each record includes the article <strong>title</strong>, <strong>content</strong>, and its <strong>category</strong></li>
        </ul>
        <strong>Note:</strong> The dataset is available for <em>non-commercial and research purposes only</em>.  
        All rights, including copyright in the original articles, belong to the <strong>BBC</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

# Technology Section
st.markdown("""
<div class='section'>
    <div class='header'>🛠️ Technology Used</div>
    <div class='content'>
        This application is built entirely with <strong>Streamlit</strong>, offering an intuitive and interactive user interface for data exploration.
        <br><br>
        Core Python libraries include:
        <ul>
            <li><strong>Pandas</strong> and <strong>NumPy</strong> – Data manipulation and numerical operations</li>
            <li><strong>Scikit-learn</strong> – Implementation of LDA, TF-IDF, and KMeans models</li>
            <li><strong>BERTopic</strong> and <strong>SentenceTransformers</strong> – Transformer-based semantic topic modeling</li>
            <li><strong>PyLDAvis</strong> and <strong>Matplotlib</strong> – Visualization of topic distributions and word importance</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# Features Section
st.markdown("""
<div class='section'>
    <div class='header'>🔍 What You Can Do with It</div>
    <div class='content'>
        Using this app, you can:
        <ul>
            <li>🧠 Perform topic modeling using LDA, TF-IDF, KMeans, or BERT</li>
            <li>📊 Visualize dominant topics and top keywords per cluster</li>
            <li>💬 Analyze semantic similarity between news articles</li>
            <li>🔎 Enter your own text to predict its likely topic</li>
        </ul>
        The application empowers users to gain insights into large text corpora quickly and interactively.
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='section'>
    <div class='header'>🎯 Ready to Explore?</div>
    <div class='content'>
        Select a topic modeling technique from the sidebar to begin exploring hidden patterns in the BBC News dataset.
        <br><br>
        Gain insights into how machine learning understands and organizes text data.
    </div>
</div>
""", unsafe_allow_html=True)
