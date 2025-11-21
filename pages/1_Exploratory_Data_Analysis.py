import streamlit as st
import pandas as pd
from pathlib import Path


# ---------------- HEADER ---------------- #
def show_header():
    st.markdown(
        """
        <div style='padding: 10px 0px 20px 0px;'>
        """,
        unsafe_allow_html=True,
    )

    st.image("parfois.png", width=500)  # ← your logo

    st.markdown(
        """
        <h1 style='font-size:30px; font-weight:400; margin-top: -10px;'>
            Similarity Detection for Fashion Retail Products
        </h1>
        <hr>
        """,
        unsafe_allow_html=True,
    )


show_header()

# ---------------- PAGE CONTENT ---------------- #

st.title("Exploratory Data Analysis")

st.write(
    """
    This page lets you explore the dataset used in the similarity search.
    You can inspect columns, basic statistics, and distributions.
    """
)

# 🔧 TODO: change this to the real path of your CSV
DATA_PATH = Path("data/parfois_embeddings.csv")  # example path

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"Could not load data from `{DATA_PATH}`. Error: {e}")
        return None

df = load_data()

if df is not None:
    st.subheader("Sample of the data")
    st.dataframe(df.head())

    st.subheader("Basic information")
    st.write(f"Number of rows: **{len(df)}**")
    st.write(f"Number of columns: **{df.shape[1]}**")

    st.subheader("Column types")
    st.write(df.dtypes)

    st.subheader("Numeric summary")
    st.write(df.describe())

    # Example: pick a column to inspect
    st.subheader("Column distribution")
    col = st.selectbox("Choose a column", df.columns)
    st.write(df[col].value_counts().head(20))
else:
    st.info("Fix the CSV path in `DATA_PATH` to see the EDA.")
