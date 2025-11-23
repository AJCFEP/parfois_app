import os
import pandas as pd
import streamlit as st

# -------------------------------------------------
# Paths (go one level up from /pages to project root)
# -------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
LOGO_PATH = os.path.join(BASE_DIR, "parfois.png")

PRODUCTS_CSV    = os.path.join(DATA_DIR, "df_product.csv")
REC_TEXT_CSV    = os.path.join(DATA_DIR, "fashion_similarity_recommendations_text.csv")

EDA_DIR   = os.path.join(BASE_DIR, "EDA image files to web")
SIM_DIR   = os.path.join(EDA_DIR, "similarity_pipeline")

# -------------------------------------------------
# Global style – SAME as app.py
# -------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
        }
        .stApp [data-testid="stImage"] img {
            margin-bottom: 0.1rem;
        }
        h1, h2, h3 {
            margin-top: 0.2rem !important;
            margin-bottom: 0.2rem !important;
        }
        hr {
            margin-top: 0.2rem !important;
            margin-bottom: 0.2rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Data loaders
# -------------------------------------------------
@st.cache_data
def load_products():
    if os.path.exists(PRODUCTS_CSV):
        return pd.read_csv(PRODUCTS_CSV, low_memory=False)
    return pd.DataFrame()


@st.cache_data
def load_text_recs():
    if os.path.exists(REC_TEXT_CSV):
        return pd.read_csv(REC_TEXT_CSV)
    return pd.DataFrame()


df_products = load_products()
rec_text_df = load_text_recs()

# -------------------------------------------------
# Header: EXACTLY like app.py
# -------------------------------------------------
col_logo, col_title = st.columns([2, 3])

with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.write("parfois.png not found.")

with col_title:
    st.markdown(
        """
        <div style="
            font-family:Arial;
            font-size:26px;
            color:#555;
            margin-top:2.2rem;
            margin-bottom:0.2rem;
        ">
            Similarity Detection for Fashion Retail Products
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------------------------------
# Page title & introduction
# -------------------------------------------------
st.markdown(
    """
    # Text Similitude

    This page explains how **text-based similarity** is computed in the
    project: a brief historical perspective, how **Transformer models**
    such as BERT and **Sentence-BERT** are used, and how this method is
    applied specifically to the PARFOIS product catalogue.
    """
)

# -------------------------------------------------
# 1. Brief history of text similarity
# -------------------------------------------------
st.subheader("1. From bag-of-words to Transformer embeddings")

st.markdown(
    """
    Historically, text similarity started with **bag-of-words** and
    **TF–IDF** representations:

    - Each document (or product description) was represented by counts of
      words.
    - Similarity was computed with cosine similarity over these sparse
      vectors.
    - These methods ignore word order and deeper semantics.

    Then came **word embeddings** (e.g. Word2Vec, GloVe), where each
    word is mapped to a dense vector. This improved semantic modelling,
    but document-level similarity was still limited (often averaging
    word vectors).

    Modern approaches rely on **Transformer-based models**:

    - **BERT** (Bidirectional Encoder Representations from Transformers)
      introduced contextual embeddings: the representation of each word
      depends on its surrounding context.
    - However, plain BERT is not directly optimised for sentence-level
      similarity.

    To solve this, **Sentence-BERT (SBERT)** and related models were
    introduced:

    - They use a **Siamese / triplet architecture** on top of BERT-like
      encoders.
    - They are trained specifically for **semantic similarity** tasks,
      producing sentence embeddings that work well with cosine similarity.
    """
)

# -------------------------------------------------
# 2. How text similarity is computed in this project
# -------------------------------------------------
st.subheader("2. Text similarity pipeline in this project")

st.markdown(
    """
    In this case study, text similarity between PARFOIS products follows
    these steps:

    1. **Text field construction**
       - For each product, several metadata fields (e.g. product name,
         description, category, colour) are concatenated into a single
         textual representation.

    2. **Sentence-BERT style encoding**
       - The combined text is encoded with a **Sentence-Transformer**
         model (a BERT-like Transformer trained for sentence similarity).
       - The output is a dense embedding vector (e.g. 384 dimensions)
         for each product.

    3. **L2 normalization**
       - All embedding vectors are **L2-normalised** so that their
         length is 1.
       - This guarantees that the dot product equals **cosine similarity**.

    4. **Cosine similarity and neighbours**
       - A similarity matrix is built by computing the dot product
         between every pair of products.
       - For each product, the Top-4 most similar products (excluding
         itself) are selected.
       - The results are saved to
         `fashion_similarity_recommendations_text.csv`, which is the
         text-based recommendation file used by the Streamlit app.

    5. **Usage in the application**
       - When the user selects the **“Text only”** mode in the
         Similarity Explorer, the app uses this text-based recommendation
         file to show similar products and their cosine similarity scores.
    """
)

# -------------------------------------------------
# 3. Example – inspecting text-based neighbours
# -------------------------------------------------
st.subheader("3. Example: similar products by text")

if df_products.empty or rec_text_df.empty:
    st.info(
        "Product table or text-based recommendation file not found. "
        "Please ensure df_product.csv and fashion_similarity_recommendations_text.csv "
        "are available in the data folder."
    )
else:
    df_local = df_products.copy()

    # Ensure product_id exists (same convention as main app)
    if "product_id" not in df_local.columns:
        if "PROD_CLR" in df_local.columns:
            df_local["product_id"] = df_local["PROD_CLR"]
        else:
            st.error(
                "df_product.csv has no 'product_id' or 'PROD_CLR' column. "
                "Cannot align text recommendations."
            )
            st.stop()

    valid_ids = set(rec_text_df["product_id"])
    df_valid = df_local[df_local["product_id"].isin(valid_ids)].copy()

    if df_valid.empty:
        st.warning("No overlap between df_product and text recommendation product_ids.")
    else:
        def make_label(row):
            return (
                f"{row['PROD_COD']} | {row['CLR_DES']} | "
                f"{row['product_id']} | {str(row['PROD_DES'])[:50]}"
            )

        df_valid["label"] = df_valid.apply(make_label, axis=1)
        df_valid = df_valid.sort_values("label")

        label_to_pid = dict(zip(df_valid["label"], df_valid["product_id"]))

        selected_label = st.selectbox(
            "Choose a product to inspect (text similarity):",
            options=list(label_to_pid.keys()),
        )

        selected_pid = label_to_pid[selected_label]
        row_sel = df_valid[df_valid["product_id"] == selected_pid].iloc[0]

        st.markdown("**Selected product (metadata)**")
        st.write(
            {
                "product_id": selected_pid,
                "PROD_COD": row_sel.get("PROD_COD"),
                "PROD_DES": row_sel.get("PROD_DES"),
                "CLR_DES": row_sel.get("CLR_DES"),
            }
        )

        # Get its text-based recommendations
        rec_row = rec_text_df[rec_text_df["product_id"] == selected_pid]
        if rec_row.empty:
            st.info("No text-based neighbours found for this product.")
        else:
            rec_row = rec_row.iloc[0]
            sim_ids = [rec_row[f"sim_{i}_id"] for i in range(1, 5)]
            sim_scores = [rec_row[f"sim_{i}_score"] for i in range(1, 5)]

            rows = []
            for pid_sim, score in zip(sim_ids, sim_scores):
                r = df_valid[df_valid["product_id"] == pid_sim]
                if not r.empty:
                    r = r.iloc[0]
                    rows.append(
                        {
                            "neighbor_product_id": pid_sim,
                            "similarity_score": score,
                            "PROD_COD": r.get("PROD_COD"),
                            "PROD_DES": r.get("PROD_DES"),
                            "CLR_DES": r.get("CLR_DES"),
                        }
                    )

            st.markdown("**Top-4 neighbours based on text similarity**")
            if rows:
                st.dataframe(pd.DataFrame(rows))
            else:
                st.info("No detailed metadata found for neighbours.")

# (Optional) if you later export any text-specific diagnostic image,
# you can show it here with os.path.exists checks from SIM_DIR.
