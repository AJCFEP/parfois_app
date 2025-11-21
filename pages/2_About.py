import streamlit as st
import os

# -------------------------------------------------
# Paths (go one level up from /pages to project root)
# -------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "parfois.png")

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
# About page content (your previous text)
# -------------------------------------------------
st.markdown(
    """
    # About this Project

    This web application is part of a **group project** developed in the
    **Master in Data Analytics (MADSAD)** at **FEP – University of Porto**,
    within the course **“Quantitative Case Studies”**.

    The goal of the project is to study and implement a system for
    **fashion product similarity detection**, using a real catalogue of
    PARFOIS products and modern data science tools.
    """
)

# (... keep all your previous sections here: notebook description,
#     web app description, deployment, team members, references, etc.)

st.markdown("### Project Pipeline Overview")
st.image(
    os.path.join(BASE_DIR, "workflow.png"),
    caption="High-level workflow of the Fashion Product Similarity project",
    width=600,
)
