import streamlit as st
import pandas as pd
import os

# -------------------------------------------------
# Paths (same logic as other pages)
# -------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
LOGO_PATH  = os.path.join(BASE_DIR, "parfois.png")

REC_CSV    = os.path.join(DATA_DIR, "fashion_similarity_recommendations.csv")

EDA_DIR    = os.path.join(BASE_DIR, "EDA image files to web")
SIM_DIR    = os.path.join(EDA_DIR, "similarity_pipeline")

# optional: folder with example grids created by show_product_and_matches
# e.g., example_matches_145869_BL.png, example_matches_222875_TU.png

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
def load_rec_and_stats():
    rec_df = pd.read_csv(REC_CSV)
    stats_path = os.path.join(SIM_DIR, "similarity_score_summary.csv")
    stats_df = pd.read_csv(stats_path) if os.path.exists(stats_path) else None
    return rec_df, stats_df

rec_df, stats_df = load_rec_and_stats()

@st.cache_data
def load_products():
    products_csv = os.path.join(DATA_DIR, "df_product.csv")
    return pd.read_csv(products_csv, low_memory=False)

df_products = load_products()

@st.cache_data
def build_image_map(images_root: str):
    """
    Scan the images folder and build mapping: product_id -> list of image paths.
    Assumes filenames like 140486_BM_1.jpg  -> product_id = '140486_BM'.
    """
    mapping = {}
    for root, _, files in os.walk(images_root):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                parts = fname.split("_")
                if len(parts) >= 2:
                    product_id = "_".join(parts[:2])
                else:
                    product_id = os.path.splitext(fname)[0]
                full_path = os.path.join(root, fname)
                mapping.setdefault(product_id, []).append(full_path)
    return mapping

IMAGES_DIR = os.path.join(BASE_DIR, "images")
image_map = build_image_map(IMAGES_DIR)

# -------------------------------------------------
# Header: EXACTLY like app.py
# -------------------------------------------------
col_logo, col_title = st.columns([2, 3])

with col_logo:
    st.image(LOGO_PATH, use_container_width=True)

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
# Page title
# -------------------------------------------------
st.markdown(
    """
    <div style="font-size:32px; font-weight:600;
                margin-top:4px; margin-bottom:4px;">
        Similarity Pipeline & Scores
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    """
    This page explains the **end-to-end pipeline** used to compute
    similarity between PARFOIS products, from images to the
    `fashion_similarity_recommendations.csv` file used by the app.
    """
)

# -------------------------------------------------
# 1. Pipeline diagram / explanation
# -------------------------------------------------
st.subheader("1. Pipeline overview")

st.markdown(
    """
```text
Product images + metadata
           |
           v
     CLIP image encoder
   (ViT-B/32, pretrained)
           |
           v
  512-dim image embeddings
           |
           v
  L2 normalization of vectors
           |
           v
Cosine similarity (dot product)
           |
           v
Top-4 nearest neighbours per product
  -> fashion_similarity_recommendations.csv
           |
           v
Streamlit app:
 - Main page: product recommendations
 - This page: diagnostics & score analysis
