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

# Root EDA folder and subfolders as used in the notebook
EDA_DIR            = os.path.join(BASE_DIR, "EDA image files to web")
PRODUCT_STRUCT_DIR = os.path.join(EDA_DIR, "product_structure")
SALES_STRUCT_DIR   = os.path.join(EDA_DIR, "sales_structure")
PCA_DIR            = os.path.join(EDA_DIR, "pca")
CLUSTER_DIR        = os.path.join(EDA_DIR, "clustering")

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

# EDA CSV tables
@st.cache_data
def load_eda_tables():
    tables = {}
    def safe_read(path):
        return pd.read_csv(path) if os.path.exists(path) else None

    tables["prod_attr"]  = safe_read(os.path.join(EDA_DIR, "df_products_attribute_summary.csv"))
    tables["prod_num"]   = safe_read(os.path.join(EDA_DIR, "df_products_numeric_summary.csv"))
    tables["sales_attr"] = safe_read(os.path.join(EDA_DIR, "df_sales_attribute_summary.csv"))
    tables["sales_num"]  = safe_read(os.path.join(EDA_DIR, "df_sales_numeric_summary.csv"))
    tables["pca_var"]    = safe_read(os.path.join(PCA_DIR, "pca_explained_variance.csv"))
    tables["kmeans_counts"] = safe_read(os.path.join(CLUSTER_DIR, "kmeans_cluster_counts.csv"))
    return tables

eda_tables = load_eda_tables()

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
    This page combines **live summaries** of the PARFOIS datasets with
    **static figures and tables** exported from the original EDA notebook,
    including PCA and clustering diagnostics.
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
# Tabs for notebook EDA results
# -------------------------------------------------
st.markdown("## Notebook EDA Results")

tab_prod, tab_sales, tab_corr, tab_sales_rel, tab_pca, tab_cluster = st.tabs(
    ["Products structure", "Sales structure", "Correlations",
     "Sales relationships", "PCA", "Clustering"]
)

# ---------- Products structure ----------
with tab_prod:
    st.subheader("Attribute types – df_products")
    st.image(
        os.path.join(EDA_DIR, "df_products_dtype_counts.png"),
        caption="Count of attribute types – df_products",
        #use_column_width=True,
        width=700,
    )

    st.subheader("Attribute type summaries – df_products")
    st.image(
        os.path.join(EDA_DIR, "df_products_attribute_3tables.png"),
        caption="Attribute type summary – df_products (3 blocks)",
        use_column_width=True,
        #width=300,
    )

    if eda_tables["prod_attr"] is not None:
        with st.expander("Show attribute summary table (CSV)", expanded=False):
            st.dataframe(eda_tables["prod_attr"])

    with st.expander("Show blocks 1–3 separately (PNG)", expanded=False):
        st.image(
            os.path.join(EDA_DIR, "df_products_attr_summary_block1.png"),
            caption="df_products – block 1",
            use_column_width=True,
            #width=300,
        )
        st.image(
            os.path.join(EDA_DIR, "df_products_attr_summary_block2.png"),
            caption="df_products – block 2",
            use_column_width=True,
            #width=300,
        )
        st.image(
            os.path.join(EDA_DIR, "df_products_attr_summary_block3.png"),
            caption="df_products – block 3",
            use_column_width=True,
            #width=300,
        )

    st.subheader("Numeric summary – df_products (from notebook)")
    st.image(
        os.path.join(EDA_DIR, "df_products_numeric_summary.png"),
        caption="Numeric summary – df_products",
        #use_column_width=True,
        width=300,
    )

    if eda_tables["prod_num"] is not None:
        with st.expander("Show numeric summary table (CSV)", expanded=False):
            st.dataframe(eda_tables["prod_num"])

    # New: products numeric boxplots without BAR_COD
    st.subheader("Boxplots – selected numeric attributes (excluding BAR_COD)")
    prod_box_path = os.path.join(PRODUCT_STRUCT_DIR, "products_numeric_boxplots.png")
    if os.path.exists(prod_box_path):
        st.image(prod_box_path, use_column_width=True)
    else:
        st.info("Boxplot image not found. Regenerate it from the notebook.")

# ---------- Sales structure ----------
with tab_sales:
    st.subheader("Attribute types – df_sales")
    st.image(
        os.path.join(EDA_DIR, "df_sales_dtype_counts.png"),
        caption="Count of attribute types – df_sales",
        use_column_width=True,
        #width=300,
    )

    st.subheader("Attribute type summary – df_sales")
    st.image(
        os.path.join(EDA_DIR, "df_sales_attr_summary_block1.png"),
        caption="Attribute type summary – df_sales",
        use_column_width=True,
        #width=300,
    )

    if eda_tables["sales_attr"] is not None:
        with st.expander("Show sales attribute summary table (CSV)", expanded=False):
            st.dataframe(eda_tables["sales_attr"])

    st.subheader("Numeric summary – df_sales (from notebook)")
    st.image(
        os.path.join(EDA_DIR, "df_sales_numeric_summary.png"),
        caption="Numeric summary – df_sales",
        use_column_width=True,
        #width=300,
    )

    if eda_tables["sales_num"] is not None:
        with st.expander("Show numeric summary table – sales (CSV)", expanded=False):
            st.dataframe(eda_tables["sales_num"])

    # New: sales boxplots
    st.subheader("Boxplots – sales metrics")
    col1, col2 = st.columns(2)
    qty_path = os.path.join(SALES_STRUCT_DIR, "sales_qty_boxplot.png")
    amt_path = os.path.join(SALES_STRUCT_DIR, "sales_amt_fx_rate_boxplot.png")

    with col1:
        if os.path.exists(qty_path):
            st.image(qty_path, caption="SALES_QTY boxplot", use_column_width=True)
        else:
            st.info("SALES_QTY boxplot not found.")

    with col2:
        if os.path.exists(amt_path):
            st.image(amt_path, caption="SALES_AMT_FX_RATE boxplot", use_column_width=True)
        else:
            st.info("SALES_AMT_FX_RATE boxplot not found.")

# ---------- Correlations ----------
with tab_corr:
    st.subheader("Pearson correlation heatmap")
    st.image(
        os.path.join(EDA_DIR, "pearson_correlation_heatmap.png"),
        caption="Pearson correlation heatmap (quantitative features)",
        #use_column_width=True,
        width=500,
    )

    st.subheader("Spearman correlation heatmap")
    st.image(
        os.path.join(EDA_DIR, "spearman_correlation_heatmap.png"),
        caption="Spearman correlation heatmap (quantitative features)",
        #use_column_width=True,
        width=500,
    )

# ---------- Sales relationships ----------
with tab_sales_rel:
    st.subheader("Pairplot – key sales variables")
    st.image(
        os.path.join(EDA_DIR, "sales_pairplot.png"),
        caption="Pairplot of SALES_QTY and SALES_AMT_FX_RATE",
        #use_column_width=True,
        width=300,
    )

# ---------- PCA ----------
with tab_pca:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("PCA – variance explained")
        scree_path = os.path.join(PCA_DIR, "pca_scree_plot.png")
        if os.path.exists(scree_path):
            st.image(scree_path, caption="PCA – scree plot", use_column_width=True)
        else:
            st.info("PCA scree plot not found. Regenerate it from the notebook.")

    with col2:
        st.subheader("PCA – PC1 vs PC2 scatter")
        scatter_path = os.path.join(PCA_DIR, "pca_pc1_pc2_scatter.png")
        if os.path.exists(scatter_path):
            st.image(scatter_path, caption="PC1 vs PC2 scatter", use_column_width=True)
        else:
            st.info("PC1 vs PC2 scatter plot not found.")

    st.subheader("PCA variance table")
    if eda_tables["pca_var"] is not None:
        st.dataframe(eda_tables["pca_var"])
    else:
        st.info("pca_explained_variance.csv not found in PCA folder.")


# ---------- Clustering ----------
with tab_cluster:
    st.subheader("K-Means diagnostics")

    elbow_path = os.path.join(CLUSTER_DIR, "kmeans_elbow_curve.png")
    sil_path   = os.path.join(CLUSTER_DIR, "kmeans_silhouette_scores.png")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(elbow_path):
            st.image(elbow_path, caption="Elbow curve", use_column_width=True)
        else:
            st.info("Elbow curve image not found.")
    with col2:
        if os.path.exists(sil_path):
            st.image(sil_path, caption="Silhouette scores", use_column_width=True)
        else:
            st.info("Silhouette scores image not found.")

    if eda_tables["kmeans_counts"] is not None:
        st.subheader("Cluster size distribution (k-means)")
        st.dataframe(eda_tables["kmeans_counts"])
    else:
        st.info("kmeans_cluster_counts.csv not found in clustering folder.")
