import os
import pandas as pd
import streamlit as st

# -------------------------------------------------
# Paths (go one level up from /pages to project root)
# -------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
LOGO_PATH  = os.path.join(BASE_DIR, "parfois.png")

PRODUCTS_CSV  = os.path.join(DATA_DIR, "df_product.csv")
REC_IMAGE_CSV = os.path.join(DATA_DIR, "fashion_similarity_recommendations_clip.csv")

EDA_DIR    = os.path.join(BASE_DIR, "EDA image files to web")
SIM_DIR    = os.path.join(EDA_DIR, "similarity_pipeline")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

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
def load_image_recs():
    if os.path.exists(REC_IMAGE_CSV):
        return pd.read_csv(REC_IMAGE_CSV)
    return pd.DataFrame()


@st.cache_data
def build_image_map(images_root: str):
    """
    Scan the images folder and build mapping: product_id -> list of image paths.
    Assumes filenames like 140486_BM_1.jpg -> product_id = '140486_BM'.
    """
    mapping = {}
    if not os.path.exists(images_root):
        return mapping

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


df_products  = load_products()
rec_image_df = load_image_recs()
image_map    = build_image_map(IMAGES_DIR)

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
    # Image Similarity (CLIP)

    This page explains how **image-based similarity** is computed using
    **CLIP (Contrastive Language–Image Pretraining)**, and how it is
    used in this project to find visually similar PARFOIS products.
    """
)

# -------------------------------------------------
# 1. Brief history of visual features and CLIP
# -------------------------------------------------
st.subheader("1. From CNN features to CLIP")

st.markdown(
    """
    Early visual similarity systems relied on:

    - **Hand-crafted features** (e.g. SIFT, HOG) to describe shapes,
      edges and textures.
    - Later, **Convolutional Neural Networks (CNNs)** were used to
      extract high-level visual features (e.g. from ImageNet-pretrained
      networks such as VGG, ResNet).

    These approaches produced powerful visual embeddings, but they were
    typically trained purely on images, without directly connecting them
    to language.

    **CLIP (Contrastive Language–Image Pretraining)** introduced a
    different idea:

    - Train an image encoder and a text encoder jointly on a very large
      dataset of (image, text) pairs.
    - Use a **contrastive objective** so that matching image–text pairs
      have similar embeddings, while mismatched pairs are far apart.
    - After training, the image encoder alone can be used as a strong
      visual feature extractor, aligned with semantic concepts.
    """
)

# -------------------------------------------------
# 2. Image similarity pipeline in this project
# -------------------------------------------------
st.subheader("2. How CLIP-based similarity is computed here")

st.markdown(
    """
    In this case study, image similarity between products follows these steps:

    1. **Image collection**
       - Each product is associated with one or more image files
         (e.g. product catalogue photos) stored in the `images/` folder.
       - Filenames encode the `product_id` (e.g. `140486_BM_1.jpg`).

    2. **CLIP image encoder (ViT-B/32)**
       - A pretrained CLIP model with a Vision Transformer backbone
         (ViT-B/32) is loaded.
       - For each product, **one representative image** is passed through
         the image encoder to obtain a 512-dimensional embedding.

    3. **L2 normalization**
       - The CLIP embeddings are L2-normalised so that their length is 1.
       - This allows the **dot product to be interpreted as cosine similarity**.

    4. **Cosine similarity and neighbours**
       - A similarity matrix is built by computing the dot product
         between each pair of product image embeddings.
       - For each product, the Top-4 most similar products (excluding
         itself) are selected.
       - The results are saved to
         `fashion_similarity_recommendations_clip.csv`, which is the
         image-based recommendation file used by the app.

    5. **Usage in the application**
       - When the user selects the **“Image only (CLIP)”** mode in the
         Similarity Explorer, the app uses this file to display visually
         similar products and their similarity scores.
    """
)

# -------------------------------------------------
# 2.1 Diagram – CLIP image similarity pipeline
# -------------------------------------------------
st.subheader("Diagram – Image similarity pipeline (CLIP)")

clip_diag_path = os.path.join(SIM_DIR, "clip_image_similarity_pipeline.png")
if os.path.exists(clip_diag_path):
    st.image(
        clip_diag_path,
        caption="Image similarity pipeline (product images → CLIP → cosine similarity).",
        use_column_width=True,
    )
else:
    st.info(
        "clip_image_similarity_pipeline.png not found in similarity_pipeline folder. "
        "Place the diagram in 'EDA image files to web/similarity_pipeline'."
    )

# -------------------------------------------------
# 3. Example – visually similar products (CLIP)
# -------------------------------------------------
st.subheader("3. Example: visually similar products with CLIP")

if df_products.empty or rec_image_df.empty:
    st.info(
        "Product table or CLIP-based recommendation file not found. "
        "Please ensure df_product.csv and fashion_similarity_recommendations_clip.csv "
        "are available in the data folder."
    )
else:
    df_local = df_products.copy()

    # Ensure product_id exists
    if "product_id" not in df_local.columns:
        if "PROD_CLR" in df_local.columns:
            df_local["product_id"] = df_local["PROD_CLR"]
        else:
            st.error(
                "df_product.csv has no 'product_id' or 'PROD_CLR' column. "
                "Cannot align CLIP recommendations."
            )
            st.stop()

    valid_ids = set(rec_image_df["product_id"])
    df_valid = df_local[df_local["product_id"].isin(valid_ids)].copy()

    if df_valid.empty:
        st.warning("No overlap between df_product and CLIP recommendation product_ids.")
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
            "Choose a product to inspect (image similarity):",
            options=list(label_to_pid.keys()),
        )

        selected_pid = label_to_pid[selected_label]
        row_sel = df_valid[df_valid["product_id"] == selected_pid].iloc[0]
        paths_sel = image_map.get(selected_pid, [])

        col_sel, col_sims = st.columns([1, 3])

        with col_sel:
            st.markdown("**Selected product**")
            st.write(
                {
                    "product_id": selected_pid,
                    "PROD_COD": row_sel.get("PROD_COD"),
                    "PROD_DES": row_sel.get("PROD_DES"),
                    "CLR_DES": row_sel.get("CLR_DES"),
                }
            )
            if paths_sel:
                st.image(paths_sel[0], caption=f"{selected_pid}", width=280)
            else:
                st.info("No image found for this product in the images folder.")

        with col_sims:
            st.markdown("**Top-4 visually similar products (CLIP)**")

            rec_row = rec_image_df[rec_image_df["product_id"] == selected_pid]
            if rec_row.empty:
                st.info("No CLIP-based neighbours found for this product.")
            else:
                rec_row = rec_row.iloc[0]
                sim_ids = [rec_row[f"sim_{i}_id"] for i in range(1, 5)]
                sim_scores = [rec_row[f"sim_{i}_score"] for i in range(1, 5)]

                sim_cols = st.columns(4)
                for col, pid_sim, score in zip(sim_cols, sim_ids, sim_scores):
                    with col:
                        col.markdown(f"**{pid_sim}**")
                        col.caption(f"Similarity: {score:.3f}")

                        r = df_valid[df_valid["product_id"] == pid_sim]
                        if not r.empty:
                            r = r.iloc[0]
                            col.write(r.get("PROD_DES"))
                            col.caption(r.get("CLR_DES"))

                        paths = image_map.get(pid_sim, [])
                        if paths:
                            col.image(paths[0], use_column_width=True)
                        else:
                            col.info("No image")
                            # -------------------------------------------------
# 4. References
# -------------------------------------------------
st.subheader("References to go further on the subject")

st.markdown(
    """
- Radford, A., Kim, J. W., Hallacy, C., et al. (2021).  
  *Learning transferable visual models from natural language supervision (CLIP).*

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021).  
  *An image is worth 16×16 words: Transformers for image recognition at scale (ViT).*

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).  
  *ImageNet classification with deep convolutional neural networks.*

- Goodfellow, I., Bengio, Y., & Courville, A. (2016).  
  *Deep Learning.* MIT Press. (Chapters on CNNs and representation learning)
"""
)

