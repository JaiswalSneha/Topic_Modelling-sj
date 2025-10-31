import streamlit as st

# Define pages (by file or function)

page2 = st.Page("data_preprocess.py", title="Home Page", icon="🏠")   
page3 = st.Page("lda_bow.py", title="LDA with BOW", icon="🧮")        
page4 = st.Page("predict.py", title="Prediction", icon="🎯")          
page5 = st.Page("lda_tfidf.py", title="LDA with TF-IDF", icon="📊")   
page6 = st.Page("kmeans.py", title="KMeans TM", icon="🔵")            
page7 = st.Page("bert.py", title="BERT TM", icon="🤖")                
page1 = st.Page("about_me.py", title="About Me", icon="👩‍💻")           


# Create navigation
pg = st.navigation([page2, page3, page4, page5, page6, page7,page1])

# Run navigation
pg.run()