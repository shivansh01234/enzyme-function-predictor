🧬 Genomic NLP: Enzyme Function Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_LINK_HERE)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Bioinformatics](https://img.shields.io/badge/Bioinformatics-Biopython-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20SVM-orange.svg)
![Deployment](https://img.shields.io/badge/Deployed%20on-Streamlit-red.svg)

> A bioinformatics web application that predicts enzyme functional classes (EC numbers) directly from raw amino acid sequences using Natural Language Processing.

<br>

## 📖 Project Overview
Understanding enzyme function is a foundational task in industrial biotechnology and genomics. This project treats biological sequences as a mathematical language, utilizing Natural Language Processing (NLP) techniques to predict the functional family of an enzyme strictly from its raw amino acid sequence. 

The pipeline tokenizes protein sequences using a 4-mer sliding window algorithm and translates them into high-dimensional mathematical matrices via TF-IDF vectorization. A trained machine learning classification engine then evaluates this data to categorize the protein into one of the six primary enzyme classes (EC 1 through EC 6). The engine is deployed as a live, interactive web application using Streamlit.

## ✨ Core Features
* **Interactive Sequence Analysis:** Users can paste raw, unformatted FASTA sequences directly into the browser for real-time analysis.
* **Biological NLP Translation:** Utilizes advanced text-vectorization (`TfidfVectorizer`) combined with a $k$-mer sliding window to extract deep structural motifs.
* **Multi-Algorithm Optimization:** The underlying architecture was tested across Random Forest, Support Vector Machines (Linear SVC), and XGBoost to find the absolute mathematical ceiling for sequence-based prediction.
* **Rapid Diagnostic Output:** Instantly outputs the predicted Enzyme Commission (EC) class alongside its specific biological function (e.g., Hydrolase, Ligase, etc.).

## 🛠️ Tech Stack & Architecture

| Phase | Technologies Used | Purpose |
| :--- | :--- | :--- |
| **Data Processing** | Biopython, Pandas | Parsing massive `.fasta` files and creating balanced dataframes |
| **Feature Extraction** | Scikit-Learn (TF-IDF) | Translating raw amino acids into scaled mathematical sparse matrices |
| **Model Engineering** | Scikit-Learn, XGBoost | Training ensemble and gradient-boosted classifiers |
| **Web Deployment** | Streamlit, Joblib | Model serialization and front-end interface hosting |

## 🧠 The Machine Learning Workflow

1. **Data Acquisition & Balancing:** Raw FASTA files containing thousands of protein sequences were parsed and heavily sub-sampled to create a perfectly balanced dataset, preventing class bias in the AI.
2. **$k$-mer Extraction:** Chopping continuous protein sequences into overlapping 4-mers (e.g., `MVLS`, `VLSP`) to simulate biological "words".
3. **TF-IDF Vectorization:** Translating the 4-mers into a matrix, mathematically amplifying rare, class-defining structural motifs while penalizing common biological background noise.
4. **Model Training & Serialization:** Training the classifier on the sparse matrix, mapping confusion matrices to diagnose blindspots, and saving the final optimized pipeline as `.pkl` files for instant browser inference.

## 💻 How to Run Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/shivansh01234/enzyme-function-predictor.git](https://github.com/shivansh01234/enzyme-function-predictor.git)
