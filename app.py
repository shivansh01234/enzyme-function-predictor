import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Enzyme NLP Diagnostic Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Background Setup (Load Models) ---
@st.cache_resource
def load_setup():
    # Load the NLP translator and the machine learning model
    try:
        model = joblib.load('enzyme_model.pkl')
        cv = joblib.load('kmer_translator.pkl')
        models_loaded = True
    except FileNotFoundError:
        model, cv = None, None
        models_loaded = False
        
    return model, cv, models_loaded

model, cv, models_loaded = load_setup()

# --- Biological Dictionary ---
enzyme_classes = {
    0: ("EC 1: Oxidoreductases", "Catalyze oxidation/reduction reactions (transfer of electrons)."),
    1: ("EC 2: Transferases", "Transfer a functional group (e.g., a methyl or phosphate group) from one molecule to another."),
    2: ("EC 3: Hydrolases", "Catalyze the hydrolysis of various bonds (cleaving bonds with water)."),
    3: ("EC 4: Lyases", "Cleave various bonds by means other than hydrolysis and oxidation."),
    4: ("EC 5: Isomerases", "Catalyze isomerization changes within a single molecule."),
    5: ("EC 6: Ligases", "Join two molecules with covalent bonds.")
}

def get_kmers(sequence, size=4):
    """Chops a raw sequence into overlapping 4-mers."""
    clean_seq = re.sub(r'[^a-zA-Z]', '', sequence).lower()
    kmers = [clean_seq[x:x+size] for x in range(len(clean_seq) - size + 1)]
    return ' '.join(kmers)

# --- Sidebar Navigation ---
st.sidebar.title("🧬 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("Select a Section:", ["Project Overview", "Data & NLP Evaluation", "Live Diagnostic Engine"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**Engineer:** Shivansh Sahu\n\n"
    "**Domain:** Bioinformatics & Natural Language Processing\n\n"
    "**Goal:** Translating raw genomic data into functional enzyme classifications using classical machine learning and biological NLP."
)

# --- PAGE 1: Project Overview ---
if page == "Project Overview":
    st.title("Genomic NLP: Enzyme Function Predictor")
    st.markdown("### Translating Amino Acids into Functional Diagnostics")
    
    st.write("""
    Understanding enzyme function is a foundational task in industrial biotechnology and genomics. 
    Traditionally, discovering an enzyme's functional class requires complex 3D folding simulations or 
    wet-lab assays.
    
    This machine learning project treats biological sequences as a mathematical language. By utilizing 
    Natural Language Processing (NLP) techniques, this algorithm acts as a digital diagnostic assistant, 
    predicting the functional family of an enzyme strictly from its raw amino acid string.
    """)
    
    st.info("**Technical Highlights:**\n"
            "* **Architecture:** Pipeline combining Term Frequency-Inverse Document Frequency (TF-IDF) and Classification Engines.\n"
            "* **Feature Engineering:** Tokenized continuous protein sequences using a 4-mer sliding window algorithm to simulate biological 'words'.\n"
            "* **Optimization:** Translated raw text into a 25,000-dimensional mathematical matrix to amplify rare structural motifs while penalizing background noise.")

# --- PAGE 2: Data & NLP Evaluation ---
elif page == "Data & NLP Evaluation":
    st.title("📊 Model Evaluation & NLP Analytics")
    st.write("A deep dive into the algorithmic performance and the confusion matrix diagnostics.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Clinical Confusion Matrix")
        st.write("Visualizing exactly which biological classes the AI correctly identifies, and where the $k$-mer blindspots lie.")
        # Looks for the confusion matrix we generated earlier!
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.warning("Image 'confusion_matrix.png' not found in the root folder.")
            

    with col2:
        st.subheader("2. Class Distribution")
        st.write("Ensuring the training dataset was perfectly balanced across all 6 EC families to prevent algorithmic bias.")
        if os.path.exists("class_distribution.png"):
            st.image("class_distribution.png", use_container_width=True)
        else:
            st.info("Class distribution chart placeholder.")
            
        st.subheader("3. K-Mer Frequency Analysis")
        st.write("Mapping the most heavily weighted 4-mers (TF-IDF scores) that define specific enzyme active sites.")
        if os.path.exists("kmer_frequencies.png"):
            st.image("kmer_frequencies.png", use_container_width=True)
        else:
            st.info("K-Mer frequency chart placeholder.")

# --- PAGE 3: Live Diagnostic Engine ---
elif page == "Live Diagnostic Engine":
    st.title("🩺 Live Sequence Predictor")
    st.write("Paste the raw amino acid sequence of any uncharacterized protein to see the NLP model's real-time prediction.")
    
    if not models_loaded:
        st.error("⚠️ AI Models not found. Please ensure 'enzyme_model.pkl' and 'kmer_translator.pkl' are in the project folder.")
    else:
        st.markdown("### Input Cellular Sequence")
        
        user_sequence = st.text_area(
            "Raw FASTA format (without the > header):",
            height=200,
            placeholder="e.g., MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
        )
        
        st.markdown("---")
        
        if st.button("Run NLP Sequence Analysis", type="primary"):
            if len(user_sequence.strip()) < 10:
                st.warning("Please enter a valid protein sequence of at least 10 amino acids.")
            else:
                with st.spinner("Translating sequence into 4-mers and processing mathematical matrix..."):
                    
                    # 1. Chop sequence into k-mers
                    kmer_sentence = get_kmers(user_sequence)
                    
                    # 2. Vectorize the text
                    vectorized_sequence = cv.transform([kmer_sentence]).toarray()
                    
                    # 3. Predict
                    prediction = model.predict(vectorized_sequence)[0]
                    predicted_name, predicted_desc = enzyme_classes[prediction]
                    
                    st.success("Analysis Complete.")
                    
                    st.subheader("Diagnostic Results:")
                    
                    colA, colB = st.columns(2)
                    with colA:
                        st.metric(label="Predicted Class", value=predicted_name.split(':')[0])
                        st.markdown(f"**Family:** {predicted_name.split(':')[1]}")
                        st.markdown(f"**Biological Function:** {predicted_desc}")
                    
                    with colB:
                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(vectorized_sequence)[0]
                            confidence = probabilities[prediction] * 100
                            st.metric(label="Algorithm Confidence", value=f"{confidence:.2f}%")
                            st.progress(int(confidence))
                        else:
                            st.metric(label="Algorithm Confidence", value="N/A")
                            st.info("Confidence metrics are calculated by ensemble models (XGBoost/RF), not Linear SVMs.")
