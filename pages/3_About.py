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
# About page content
# -------------------------------------------------
st.markdown(
    """
    # About this Project

    This web application is part of a **group project** developed in the
    **Master in Data Analytics (MADSAD)** at **FEP – University of Porto**,
    within the course **“Quantitative Case Studies”**.

    The goal of the project is to study and implement a system for
    **fashion product similarity detection**, using a real catalogue of
    PARFOIS products and modern data science tools. The final engine
    combines **text information** (product metadata) and **visual
    information** (product images) in a hybrid similarity model.
    """
)

# -------------------- From Notebook to Hybrid Similarity Engine --------------------
st.markdown(
    """
    ### From Notebook to Hybrid Similarity Engine

    The core analytical work was carried out in the Jupyter notebook  
    **`Fashion Product Similarity Detection_c.ipynb`**, which implemented
    the full pipeline from raw data to a usable similarity engine:

    1. **Data loading and cleaning**
       - Import of the PARFOIS product catalogue (IDs, names, categories,
         descriptions, prices, image references, etc.).
       - Handling of missing values, removal of duplicates, and basic text
         normalization (lowercasing, trimming, simple replacements).

    2. **Text preprocessing**
       - Creation of a consolidated textual field combining relevant
         attributes (e.g. product name, category, description).
       - Token-level cleaning (punctuation, extra spaces) to make
         the text more suitable for embedding.

    3. **Vector representations (embeddings)**
       - **Text embeddings:** a pre-trained language model
         (via the `sentence-transformers` / transformer ecosystem) converts
         each product’s text into a numerical embedding vector.
       - **Image embeddings (CLIP):** a CLIP model (ViT-B/32) converts
         representative product images into 512-dimensional visual
         embeddings.
       - Each product is thus represented by **two vectors**:
         one in a text space and one in a visual (image) space.

    4. **Similarity computation (text, image, hybrid)**
       - All embeddings are **L2-normalised**, so their length is 1 and
         the dot product corresponds to **cosine similarity**.
       - A **text-based similarity matrix** is computed from the text
         embeddings.
       - An **image-based similarity matrix** is computed from the CLIP
         embeddings.
       - A **hybrid similarity matrix** is obtained by combining the
         text and image similarities (weighted sum), capturing both
         semantic and visual proximity.

    5. **Top-k neighbours and export for the web application**
       - For each product, the system identifies the **Top-4 nearest neighbours**
         under each similarity variant:
         - Text-only
         - Image-only (CLIP)
         - Hybrid (Text + Image)
       - The results are saved into three CSV files:
         - `fashion_similarity_recommendations_text.csv`
         - `fashion_similarity_recommendations_clip.csv`
         - `fashion_similarity_recommendations_hybrid.csv`
       - Additional diagnostic outputs (score summaries and histograms) are
         also exported for the **Similarity Pipeline** page.
    """
)

st.markdown("### Project Pipeline Overview")
st.image(
    os.path.join(BASE_DIR, "workflow.png"),
    caption="High-level workflow of the Fashion Product Similarity project",
    width=300,
)

# -------------------- Web Application --------------------
st.markdown(
    """
    ### Web Application in Streamlit

    On top of the notebook, we built this interactive web interface using
    **Streamlit**, so that non-technical users can explore and interpret
    the similarity engine:

    - **Main page – Similarity Explorer**
      - Allows the user to select a product and see its Top-4 similar
        products.
      - Provides a **similarity mode selector** with three options:
        **Text only**, **Image only (CLIP)**, and **Hybrid (Text + Image)**.
      - Uses the corresponding recommendation CSV (text, image or hybrid)
        to retrieve and display neighbours and similarity scores.

    - **Exploratory Data Analysis page**
      - Provides an overview of the underlying dataset: structure of the
        product and sales tables, summary statistics, and visual
        diagnostics (boxplots, correlation heatmaps, PCA and clustering
        outputs exported from the notebook).
      - Helps to understand the coverage and structure of the catalogue
        used by the similarity engine.

    - **Similarity Pipeline & Scores page**
      - Describes the end-to-end pipeline for image and hybrid similarity.
      - Displays the distribution of similarity scores and summary metrics.
      - Allows inspection of recommendations for any product under the
        three modes (text, image, hybrid).

    - **About page (this page)**
      - Summarises the academic context, methodology, technical stack
        and bibliographic references behind the project.

    The app is built entirely in Python, using:
    - **Streamlit** for the UI and page navigation (multi-page app).
    - **pandas / NumPy / Matplotlib / seaborn** for data manipulation
      and visualisations.
    - **Transformers / sentence-transformers**, **CLIP**, **PyTorch** and
      **scikit-learn** for embeddings, dimensionality reduction,
      clustering and similarity computation (implemented in the notebook).
    """
)

# -------------------- Deployment --------------------
st.markdown(
    """
    ### GitHub and Streamlit Cloud Deployment

    To make the project reproducible and easily accessible, the code is
    hosted on **GitHub** and deployed with **Streamlit Cloud**:

    1. **GitHub repository**
       - The Jupyter notebook, the main Streamlit script (e.g. `app.py`),
         the `pages/` directory and the data / recommendation CSVs are
         stored in a GitHub repository.
       - A `requirements.txt` file lists all Python dependencies needed
         to run the project (Streamlit, scikit-learn, sentence-transformers,
         CLIP, etc.).

    2. **Connecting to Streamlit Cloud**
       - The GitHub repo is linked to Streamlit Cloud.
       - The main entry file is set to the Streamlit script that defines
         the multi-page app.
       - On each `git push`, Streamlit Cloud automatically rebuilds the
         environment and redeploys the latest version of the app.

    3. **Result**
       - Anyone with the link can interact with the similarity engine
         through the web interface, without having to install Python or
         run the notebook locally.

    Overall, this project combines **quantitative analysis**, **machine
    learning for similarity detection**, and **modern deployment tools**
    to deliver a practical case study in fashion product recommendation.
    """
)

# -------------------- Team Members --------------------
st.markdown(
    """
    ### Team Members

    This project was developed by a group of MADSAD students from FEP – UP:

    - **André Costa** – *up199401247@edu.fep.up.pt*  
    - **Catarina Monteiro** – *up202107961@edu.fep.up.pt*  
    - **João Monteiro** – *up202006793@edu.fep.up.pt*  
    - **Luis Ferreira** – *up202107032@edu.fep.up.pt*  
    - **Rodrigo Soares** – *up201602617@edu.fep.up.pt*  
    - **Telmo Barbosa** – *up201200195@edu.fep.up.pt*  
    """
)

# -------------------- References with phase selector --------------------


def show_references_by_phase():
    phases = [
        "1. Data Preparation & Exploratory Data Analysis (EDA)",
        "2. Dimensionality Reduction & Clustering",
        "3. Visual Representation Learning (CLIP)",
        "4. Similarity Computation",
        "5. Content-Based Recommender Logic",
        "6. Web Application & Deployment",
        "7. Optional: Text Similarity (Sentence-BERT)",
    ]

    refs_by_phase = {
        phases[0]: """
**References – Data Preparation & EDA**

- Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., et al. (2020). *Array programming with NumPy.* Nature, 585, 357–362.
- van der Walt, S., Colbert, S. C., & Varoquaux, G. (2011). *The NumPy array: A structure for efficient numerical computation.* Computing in Science & Engineering, 13(2), 22–30.
- McKinney, W. (2011). *pandas: A foundational Python library for data analysis and statistics.* Proceedings of the Python for Scientific Computing Conference (SciPy).
- Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment.* Computing in Science & Engineering, 9(3), 90–95.
- Waskom, M. L. (2021). *seaborn: Statistical data visualization.* Journal of Open Source Software, 6(60), 3021.
""",
        phases[1]: """
**References – Dimensionality Reduction & Clustering**

- Jolliffe, I. T. (2002). *Principal component analysis* (2nd ed.). Springer.
- Lloyd, S. P. (1982). *Least squares quantization in PCM.* IEEE Transactions on Information Theory, 28(2), 129–137.
- MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations.* Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability.
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). *Scikit-learn: Machine learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
""",
        phases[2]: """
**References – Visual Representation Learning (CLIP)**

- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., et al. (2021). *Learning transferable visual models from natural language supervision.* Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139.
- Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., et al. (2019). *PyTorch: An imperative style, high-performance deep learning library.* Advances in Neural Information Processing Systems, 32.
""",
        phases[3]: """
**References – Similarity Computation**

- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval.* Cambridge University Press.
- Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., et al. (2020). *Array programming with NumPy.* Nature, 585, 357–362.
""",
        phases[4]: """
**References – Content-Based Recommender Logic**

- Aggarwal, C. C. (2016). *Recommender systems: The textbook.* Springer.
- Lü, L., Medo, M., Yeung, C. H., Zhang, Y.-C., Zhang, Z.-K., & Zhou, T. (2012). *Recommender systems.* Physics Reports, 519(1), 1–49.
""",
        phases[5]: """
**References – Web Application & Deployment**

- Streamlit Inc. (2024). *Streamlit documentation.* Retrieved from https://docs.streamlit.io
""",
        phases[6]: """
**References – Optional Text Similarity (Sentence-BERT)**

- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence embeddings using Siamese BERT-network.* Proceedings of EMNLP-IJCNLP, 3982–3992.
"""
    }

    st.markdown("### References by Project Phase")

    selected_phase = st.selectbox(
        "Research references associated with each phase of the pipeline:",
        phases,
        index=0,
    )

    st.markdown(refs_by_phase[selected_phase])


# Call the references section
show_references_by_phase()
