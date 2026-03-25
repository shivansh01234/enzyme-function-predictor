import streamlit as st
import joblib
import re

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Enzyme Sequence Predictor",
    page_icon="🧬",
    layout="wide"
)

# --- 2. Load the Engine ---
# @st.cache_resource ensures the model only loads once, keeping the app fast.
@st.cache_resource
def load_models():
    # Load the translator and the machine learning model we saved earlier
    model = joblib.load('enzyme_model.pkl')
    cv = joblib.load('kmer_translator.pkl')
    return model, cv

try:
    model, cv = load_models()
    models_loaded = True
except FileNotFoundError:
    models_loaded = False
    st.error("⚠️ AI Models not found. Please ensure 'enzyme_model.pkl' and 'kmer_translator.pkl' are in the same folder as this script.")

# --- 3. Biological Dictionaries ---
enzyme_classes = {
    0: ("EC 1: Oxidoreductases", "Catalyze oxidation/reduction reactions (transfer of electrons)."),
    1: ("EC 2: Transferases", "Transfer a functional group (e.g., a methyl or phosphate group) from one molecule to another."),
    2: ("EC 3: Hydrolases", "Catalyze the hydrolysis of various bonds (cleaving bonds with water)."),
    3: ("EC 4: Lyases", "Cleave various bonds by means other than hydrolysis and oxidation."),
    4: ("EC 5: Isomerases", "Catalyze isomerization changes within a single molecule."),
    5: ("EC 6: Ligases", "Join two molecules with covalent bonds.")
}

# --- 4. The k-mer Translation Function ---
def get_kmers(sequence, size=4): # Updated to 4-mers!
    # Clean the input: remove any accidental spaces, numbers, or hidden characters
    clean_seq = re.sub(r'[^a-zA-Z]', '', sequence).lower()
    kmers = [clean_seq[x:x+size] for x in range(len(clean_seq) - size + 1)]
    return ' '.join(kmers)

# --- 5. Front-End UI ---
st.title("🧬 Genomic NLP: Enzyme Function Predictor")
st.markdown("### Translating raw amino acid sequences into functional classifications.")

st.write("""
This diagnostic tool utilizes Natural Language Processing (NLP) and machine learning 
to classify the functional family of a protein directly from its raw sequence.
""")

st.markdown("---")

# The Input Box
st.subheader("Input Sequence")
user_sequence = st.text_area(
    "Paste the raw amino acid sequence (FASTA format without the header):",
    height=150,
    placeholder="e.g., MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
)

# --- 6. The Prediction Logic ---
if st.button("Run Sequence Analysis", type="primary"):
    if not models_loaded:
        st.error("Cannot run analysis: Models are missing.")
    elif len(user_sequence.strip()) < 10:
        st.warning("Please enter a valid protein sequence of at least 10 amino acids.")
    else:
        with st.spinner("Translating sequence and analyzing k-mers..."):
            
            # Step A: Chop the sequence into 4-mers
            kmer_sentence = get_kmers(user_sequence)
            
            # Step B: Translate the 4-mers into the mathematical matrix
            vectorized_sequence = cv.transform([kmer_sentence]).toarray()
            
            # Step C: Ask the ML Engine for a prediction
            prediction = model.predict(vectorized_sequence)[0]
            predicted_name, predicted_desc = enzyme_classes[prediction]
            
            # Step D: Display the Results
            st.success("Sequence Analysis Complete.")
            
            st.markdown("### 📊 Diagnostic Result")
            
            # Highlight the prediction
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Class", value=predicted_name.split(':')[0])
                st.markdown(f"**Family:** {predicted_name.split(':')[1]}")
                st.markdown(f"**Biological Function:** {predicted_desc}")
            
            with col2:
                # Check if the model supports probability (Random Forest / XGBoost) or not (LinearSVC)
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(vectorized_sequence)[0]
                    confidence = probabilities[prediction] * 100
                    st.metric(label="Algorithm Confidence", value=f"{confidence:.2f}%")
                    st.progress(int(confidence))
                else:
                    st.metric(label="Algorithm Confidence", value="N/A")
                    st.info("Confidence percentages are not calculated by Linear Support Vector Machines.")