import streamlit as st

st.title("About this Project")


# -------------------------------------------------
# Header: logo (left) + subtitle (right)
# -------------------------------------------------
col_logo, col_title = st.columns([2, 3])  # tune ratio if you like

with col_logo:
    # Fill the column; the column width controls the visible size
    st.image(LOGO_PATH, use_container_width=True)

with col_title:
    # Subtitle aligned roughly vertically with the logo
    st.markdown(
        """
        <div style="
            font-family:Arial;
            font-size:26px;
            color:#555;
            margin-top:2.2rem;   /* adjust up/down to align with logo */
            margin-bottom:0.2rem;
        ">
            Similarity Detection for Fashion Retail Products
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------------------------------



# -------------------- Context --------------------
st.markdown(
    """
    ### Context

    This web application is part of a **group project** developed in the
    **Master in Data Analytics (MADSAD)** at **FEP – University of Porto**,
    within the course **“Quantitative Case Studies”**.

    The goal of the project is to study and implement a system for
    **fashion product similarity detection**, using a real catalogue of
    PARFOIS products and modern data science tools.
    """
)

# -------------------- From Notebook to Engine --------------------
st.markdown(
    """
    ### From Notebook to Similarity Engine

    The core analytical work was carried out in the Jupyter notebook  
    **`Fashion Product Similarity Detection_c.ipynb`**, which implemented
    the full pipeline from raw data to a usable similarity engine:

    1. **Data loading and cleaning**
       - Import of the PARFOIS product catalogue (IDs, names, categories,
         descriptions, prices, image links, etc.).
       - Handling of missing values, removal of duplicates, basic text
         normalization (lowercasing, trimming, simple replacements).

    2. **Text preprocessing**
       - Creation of a consolidated textual field combining relevant
         attributes (e.g. product name, category, description).
       - Simple token-level cleaning (punctuation, extra spaces) to make
         the text more suitable for embedding.

    3. **Vector representations (embeddings)**
       - Use of a **pre-trained language model** (via the
         `sentence-transformers` / transformer ecosystem) to convert each
         product’s text into a **numerical embedding vector**.
       - Each row in the dataset is now represented as a point in a
         high-dimensional space that encodes semantic similarity.

    4. **Similarity computation**
       - Construction of a **k-nearest neighbours (k-NN)** structure
         (e.g. using `scikit-learn`) based on cosine similarity between
         embeddings.
       - For a given query (product or free text), the system retrieves
         the most similar catalogue items.

    5. **Export for the web application**
       - The final dataset (including product metadata and embeddings) is
         saved to a **CSV file**.
       - This CSV is the main data source used by the Streamlit app to
         perform fast similarity lookups without recomputing embeddings
         on every request.
    """
)

# -------------------- Web Application --------------------
st.markdown(
    """
    ### Web Application in Streamlit

    On top of the notebook, we built this interactive web interface using
    **Streamlit**, so that non-technical users can explore the similarity
    engine:

    - **Main page (Similarity Search)**
      - Allows the user to search by product or by free text.
      - Loads the precomputed embeddings from the CSV.
      - Uses the k-NN structure to find and display the most similar
        products, together with their key attributes (e.g. name, price,
        category, image link).

    - **Exploratory Data Analysis page**
      - Provides an overview of the underlying dataset: number of
        products, columns, basic statistics and simple distributions.
      - Helps to understand the coverage and structure of the catalogue
        used by the similarity engine.

    - **About page (this page)**
      - Summarises the academic context, methodology and technical stack
        behind the project.

    The app is built entirely in Python, using:
    - **Streamlit** for the UI and page navigation (multi-page app).
    - **pandas / NumPy** for data manipulation.
    - **Transformers / sentence-transformers** and **scikit-learn** for
      embeddings and similarity search (implemented in the notebook).
    """
)

# -------------------- Deployment --------------------
st.markdown(
    """
    ### GitHub and Streamlit Cloud Deployment

    To make the project reproducible and easily accessible, the code is
    hosted on **GitHub** and deployed with **Streamlit Cloud**:

    1. **GitHub repository**
       - The Jupyter notebook, `app.py`, the `pages/` directory and the
         data/embeddings CSV are stored in a public or private repository.
       - A `requirements.txt` file lists all Python dependencies needed
         to run the project.

    2. **Connecting to Streamlit Cloud**
       - The GitHub repo is linked to Streamlit Cloud.
       - The main entry point is set to `app.py`.
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

    - **Student 1** – *Name to be inserted*
    - **Student 2** – *Name to be inserted*
    - **Student 3** – *Name to be inserted*
    - **Student 4** – *Name to be inserted*

    > *(full team members list.)*
    """
)

# -------------------- Pipeline Diagram (Image) --------------------
st.markdown(
    """
    ### Project Pipeline Overview
    The following diagram summarises the end-to-end workflow, from raw
    data to the deployed online application.
    """
)

# NOTE: `workflow.png` should be in the root folder of the Streamlit app (same level as app.py).
st.image(
    "workflow.png",
    caption="High-level workflow of the Fashion Product Similarity project",
    width=300 #use_column_width=True,
)

# -------------------- References --------------------
st.markdown(
    """
    ### References

    Below are some key libraries and concepts that supported this work:

    - Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence
      Embeddings using Siamese BERT-Networks.*
    - Streamlit Inc. **Streamlit Documentation** – *Build data apps in Python*.
    - Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in
      Python.* Journal of Machine Learning Research.

    > *(You can add here any additional academic articles, textbooks, or
    online resources that were consulted during the project.)*
    """
)



