import streamlit as st
import pandas as pd
import os

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

PRODUCTS_CSV = os.path.join(DATA_DIR, "df_product.csv")
SALES_CSV    = os.path.join(DATA_DIR, "df_sales.csv")
REC_CSV      = os.path.join(DATA_DIR, "fashion_similarity_recommendations.csv")

# --- Load data (basic) ---
@st.cache_data
def load_data():
    df_products = pd.read_csv(PRODUCTS_CSV, low_memory=False)
    df_sales = pd.read_csv(SALES_CSV, low_memory=False)
    rec_df = pd.read_csv(REC_CSV)
    return df_products, df_sales, rec_df

df_products, df_sales, rec_df = load_data()

# --- UI ---
st.title("PARFOIS")
st.subheader("Similarity Detection for Fashion Retail Products")

st.write("✅ Streamlit app is running.")
st.write(f"- **Products loaded:** {len(df_products)} rows")
st.write(f"- **Sales loaded:** {len(df_sales)} rows")
st.write(f"- **Recommendations loaded:** {len(rec_df)} rows")

st.write("Here are the first 5 products:")
st.dataframe(df_products.head())
