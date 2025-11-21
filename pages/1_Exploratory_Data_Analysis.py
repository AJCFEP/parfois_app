import streamlit as st
import pandas as pd
import os

# -------------------------------------------------
# Paths (go one level up from /pages to project root)
# -------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
LOGO_PATH  = os.path.join(BASE_DIR, "parfois.png")

PRODUCTS_CSV = os.path.join(DATA_DIR, "df_product.csv")
SALES_CSV    = os.path.join(DATA_DIR, "df_sales.csv")
REC_CSV      = os.path.join(DATA_DIR, "fashion_similarity_recommendations.csv")

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
# Data loaders (reusing same CSVs as main app)
# -------------------------------------------------
@st.cache_data
def load_data():
    df_products = pd.read_csv(PRODUCTS_CSV, low_memory=False)
    df_sales    = pd.read_csv(SALES_CSV, low_memory=False)
    rec_df      = pd.read_csv(REC_CSV)
    return df_products, df_sales, rec_df

df_products, df_sales, rec_df = load_data()

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
# Page title + EDA content
# -------------------------------------------------
st.markdown(
    """
    <div style="font-size:32px; font-weight:600;
                margin-top:4px; margin-bottom:4px;">
        Exploratory Data Analysis
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    """
    This page provides an overview of the dataset used in the similarity
    engine. You can inspect basic information and simple distributions.
    """
)

st.subheader("Sample of the products table")
st.dataframe(df_products.head())

st.subheader("Column types")
st.write(df_products.dtypes)

st.subheader("Numeric summary")
st.write(df_products.describe())

st.subheader("Column distribution")
col = st.selectbox("Choose a column to inspect", df_products.columns)
st.write(df_products[col].value_counts().head(30))
