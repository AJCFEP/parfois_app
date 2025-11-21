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

# Folder with exported EDA figures from the notebook
EDA_IMG_DIR = os.path.join(BASE_DIR, "EDA image files to web")

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
# Page title + basic EDA (live from dataframes)
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
    engine. You can inspect basic information and visual summaries
    exported from the original EDA notebook.
    """
)

with st.expander("📦 Sample of the products table", expanded=False):
    st.dataframe(df_products.head())

with st.expander("🔡 Column types – products", expanded=False):
    st.write(df_products.dtypes)

with st.expander("📈 Numeric summary – products (live)", expanded=False):
    st.write(df_products.describe())

with st.expander("🔍 Column distribution (choose a column)", expanded=False):
    col = st.selectbox("Choose a column", df_products.columns)
    st.write(df_products[col].value_counts().head(30))

# -------------------------------------------------
# Static EDA figures exported from the notebook
# -------------------------------------------------
st.markdown("## Notebook EDA Results")

tab_prod, tab_sales, tab_corr, tab_sales_rel = st.tabs(
    ["Products structure", "Sales structure", "Correlations", "Sales relationships"]
)

# ---------- Products ----------
with tab_prod:
    st.subheader("Attribute types – df_products")
    st.image(
        os.path.join(EDA_IMG_DIR, "df_products_dtype_counts.png"),
        caption="Count of attribute types – df_products",
        use_column_width=True,
    )

    st.subheader("Attribute type summaries – df_products")
    # 3-block large table
    st.image(
        os.path.join(EDA_IMG_DIR, "df_products_attribute_3tables.png"),
        caption="Attribute type summary – df_products (3 blocks)",
        use_column_width=True,
    )

    # individual blocks, if you want them visible too
    with st.expander("Show blocks 1–3 separately"):
        st.image(
            os.path.join(EDA_IMG_DIR, "df_products_attr_summary_block1.png"),
            caption="df_products – block 1",
            use_column_width=True,
        )
        st.image(
            os.path.join(EDA_IMG_DIR, "df_products_attr_summary_block2.png"),
            caption="df_products – block 2",
            use_column_width=True,
        )
        st.image(
            os.path.join(EDA_IMG_DIR, "df_products_attr_summary_block3.png"),
            caption="df_products – block 3",
            use_column_width=True,
        )

    st.subheader("Numeric summary – df_products (from notebook)")
    st.image(
        os.path.join(EDA_IMG_DIR, "df_products_numeric_summary.png"),
        caption="Numeric summary – df_products",
        use_column_width=True,
    )

# ---------- Sales ----------
with tab_sales:
    st.subheader("Attribute types – df_sales")
    st.image(
        os.path.join(EDA_IMG_DIR, "df_sales_dtype_counts.png"),
        caption="Count of attribute types – df_sales",
        use_column_width=True,
    )

    st.subheader("Attribute type summary – df_sales")
    st.image(
        os.path.join(EDA_IMG_DIR, "df_sales_attr_summary_block1.png"),
        caption="Attribute type summary – df_sales",
        use_column_width=True,
    )

    st.subheader("Numeric summary – df_sales (from notebook)")
    st.image(
        os.path.join(EDA_IMG_DIR, "df_sales_numeric_summary.png"),
        caption="Numeric summary – df_sales",
        use_column_width=True,
    )

# ---------- Correlations ----------
with tab_corr:
    st.subheader("Pearson correlation heatmap")
    st.image(
        os.path.join(EDA_IMG_DIR, "pearson_correlation_heatmap.png"),
        caption="Pearson Correlation Heatmap (quantitative features)",
        use_column_width=True,
    )

    st.subheader("Spearman correlation heatmap")
    st.image(
        os.path.join(EDA_IMG_DIR, "spearman_correlation_heatmap.png"),
        caption="Spearman Correlation Heatmap (quantitative features)",
        use_column_width=True,
    )

# ---------- Sales relationships ----------
with tab_sales_rel:
    st.subheader("Pairplot – key sales variables")
    st.image(
        os.path.join(EDA_IMG_DIR, "sales_pairplot.png"),
        caption="Pairplot of SALES_QTY and SALES_AMT_FX_RATE",
        use_column_width=True,
    )

