import os
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------
# Paths (go one level up from /pages to project root)
# -------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
LOGO_PATH  = os.path.join(BASE_DIR, "parfois.png")

PRODUCTS_CSV   = os.path.join(DATA_DIR, "df_product.csv")
REC_TEXT_CSV   = os.path.join(DATA_DIR, "fashion_similarity_recommendations_text.csv")
REC_IMAGE_CSV  = os.path.join(DATA_DIR, "fashion_similarity_recommendations_clip.csv")
REC_HYBRID_CSV = os.path.join(DATA_DIR, "fashion_similarity_recommendations_hybrid.csv")

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
def load_recs():
    rec_text   = pd.read_csv(REC_TEXT_CSV)   if os.path.exists(REC_TEXT_CSV)   else pd.DataFrame()
    rec_image  = pd.read_csv(REC_IMAGE_CSV)  if os.path.exists(REC_IMAGE_CSV)  else pd.DataFrame()
    rec_hybrid = pd.read_csv(REC_HYBRID_CSV) if os.path.exists(REC_HYBRID_CSV) else pd.DataFrame()
    return {
        "Text only": rec_text,
        "Image only (CLIP)": rec_image,
        "Hybrid (Text + Image)": rec_hybrid,
    }


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


@st.cache_data
def load_embeddings_for_hybrid(sim_dir: str):
    """
    Load text & image embeddings + product IDs, and build aligned matrices
    for products that have BOTH modalities.
    Returns (T, I, common_ids) or (None, None, []) if something is missing.
    """
    text_emb_path = os.path.join(sim_dir, "text_embeddings.npy")
    img_emb_path  = os.path.join(sim_dir, "clip_image_embeddings.npy")
    text_ids_path = os.path.join(sim_dir, "product_ids_text.csv")
    img_ids_path  = os.path.join(sim_dir, "product_ids_image.csv")

    if not (
        os.path.exists(text_emb_path)
        and os.path.exists(img_emb_path)
        and os.path.exists(text_ids_path)
        and os.path.exists(img_ids_path)
    ):
        return None, None, []

    try:
        text_emb = np.load(text_emb_path)
        img_emb  = np.load(img_emb_path)

        text_ids = pd.read_csv(text_ids_path)["product_id"].astype(str).tolist()
        img_ids  = pd.read_csv(img_ids_path)["product_id"].astype(str).tolist()

        text_index = {pid: i for i, pid in enumerate(text_ids)}
        img_index  = {pid: i for i, pid in enumerate(img_ids)}

        common_ids = sorted(set(text_ids) & set(img_ids))
        if not common_ids:
            return None, None, []

        T_list, I_list = [], []
        for pid in common_ids:
            T_list.append(text_emb[text_index[pid]])
            I_list.append(img_emb[img_index[pid]])

        T = np.vstack(T_list).astype("float32")
        I = np.vstack(I_list).astype("float32")
        return T, I, common_ids
    except Exception:
        # If anything goes wrong, just disable the interactive part gracefully
        return None, None, []


df_products = load_products()
rec_dfs     = load_recs()
image_map   = build_image_map(IMAGES_DIR)
T_emb, I_emb, hybrid_ids = load_embeddings_for_hybrid(SIM_DIR)

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
    # Hybrid Similarity (Text + Image)

    This page explains how the **hybrid similarity model** combines
    information from **text embeddings** and **image embeddings (CLIP)**
    to produce a more robust notion of product similarity.
    """
)

# -------------------------------------------------
# 1. Motivation and basic idea
# -------------------------------------------------
st.subheader("1. Why combine text and image similarity?")

st.markdown(
    """
    Text and image information capture **complementary aspects** of a
    product:

    - **Text (metadata)** describes the product in terms of category,
      intended usage, material, style, etc.
    - **Images** encode visual patterns such as colour, shape, texture
      and overall appearance.

    Using only one modality may miss relevant aspects:
    - Two bags described similarly in text but with very different
      colours may still be ranked as highly similar by a text-only model.
    - Two visually similar bags with slightly different descriptions may
      be ranked lower than desired by a text-only model, but clearly
      appear similar to the human eye.

    The hybrid model addresses this by combining both sources of
    information in a **single similarity score**.
    """
)

# -------------------------------------------------
# 2. How the hybrid similarity is computed
# -------------------------------------------------
st.subheader("2. Hybrid similarity pipeline in this project")

st.markdown(
    """
    The hybrid similarity follows these steps:

    1. **Independent embeddings**
       - Each product has:
         - a **text embedding** (Sentence-Transformer)  
         - an **image embedding** (CLIP ViT-B/32)
       - Both embeddings are L2-normalised.

    2. **Independent similarity matrices**
       - A text similarity matrix **S_text** is built using dot products
         between text embeddings.
       - An image similarity matrix **S_img** is built using dot
         products between CLIP embeddings.

    3. **Hybrid similarity matrix**
       - For each pair of products *(i, j)*, the hybrid similarity is:
         \\( S_{hybrid}(i,j) = \\alpha \\cdot S_{text}(i,j) + (1 - \\alpha) \\cdot S_{img}(i,j) \\)
       - In this project, \\( \\alpha \\) is initially set to **0.5**,
         giving equal weight to text and image.

    4. **Top-4 neighbours and export**
       - For each product, the Top-4 neighbours are selected according
         to **S_hybrid**.
       - The results are stored in
         `fashion_similarity_recommendations_hybrid.csv`.

    5. **Usage in the application**
       - When the user selects **“Hybrid (Text + Image)”** in the
         Similarity Explorer, this hybrid recommendation file is used.
       - This often produces results that are consistent with both the
         textual description and the visual appearance of the products.
    """
)

# -------------------------------------------------
# 2.1 Diagram – Hybrid similarity pipeline
# -------------------------------------------------
st.subheader("Diagram – Hybrid similarity pipeline")

hybrid_diag_path = os.path.join(SIM_DIR, "hybrid_similarity_pipeline.png")
if os.path.exists(hybrid_diag_path):
    st.image(
        hybrid_diag_path,
        caption="Hybrid similarity pipeline (text + image similarity combined).",
        width=500,
    )
else:
    st.info(
        "hybrid_similarity_pipeline.png not found in similarity_pipeline folder. "
        "Place the diagram in 'EDA image files to web/similarity_pipeline'."
    )

# -------------------------------------------------
# 3. Example – comparing Text vs Image vs Hybrid for one product
# -------------------------------------------------
st.subheader("3. Example: Text vs Image vs Hybrid neighbours")

if df_products.empty or all(df.empty for df in rec_dfs.values()):
    st.info(
        "Not enough data to build the hybrid example "
        "(products or recommendation files missing)."
    )
else:
    df_local = df_products.copy()

    if "product_id" not in df_local.columns:
        if "PROD_CLR" in df_local.columns:
            df_local["product_id"] = df_local["PROD_CLR"]
        else:
            st.error(
                "df_product.csv has no 'product_id' or 'PROD_CLR' column. "
                "Cannot align recommendations."
            )
            st.stop()

    # Use only products present in all three rec files
    common_ids_sets = [
        set(df["product_id"]) for df in rec_dfs.values() if not df.empty
    ]
    if not common_ids_sets:
        st.warning("Recommendation files are empty; cannot build comparison.")
    else:
        common_ids_recs = set.intersection(*common_ids_sets)
        df_valid = df_local[df_local["product_id"].isin(common_ids_recs)].copy()

        if df_valid.empty:
            st.warning("No common product_ids across text, image and hybrid recommendations.")
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
                "Choose a product to compare Text vs Image vs Hybrid:",
                options=list(label_to_pid.keys()),
            )
            selected_pid = label_to_pid[selected_label]

            row_sel = df_valid[df_valid["product_id"] == selected_pid].iloc[0]
            paths_sel = image_map.get(selected_pid, [])

            st.markdown("**Selected product**")
            col_info, col_img = st.columns([2, 1])
            with col_info:
                st.write(
                    {
                        "product_id": selected_pid,
                        "PROD_COD": row_sel.get("PROD_COD"),
                        "PROD_DES": row_sel.get("PROD_DES"),
                        "CLR_DES": row_sel.get("CLR_DES"),
                    }
                )


                
            with col_img:
                if paths_sel:
                    st.image(paths_sel[0], caption=f"{selected_pid}", width=220)
                else:
                    st.info("No image found for this product.")

            # Tabs for the three similarity modes
            tab_text, tab_image, tab_hybrid = st.tabs(
                ["Text only", "Image only (CLIP)", "Hybrid (Text + Image)"]
            )

            def build_neighbor_table(rec_df, mode_name: str):
                """
                Helper to build a neighbours table (Top-4) for a given recommendation df.
                """
                if rec_df.empty:
                    return pd.DataFrame()

                rec_row = rec_df[rec_df["product_id"] == selected_pid]
                if rec_row.empty:
                    return pd.DataFrame()

                rec_row = rec_row.iloc[0]
                sim_ids = [rec_row[f"sim_{i}_id"] for i in range(1, 5)]
                sim_scores = [rec_row[f"sim_{i}_score"] for i in range(1, 5)]

                rows = []
                for pid_sim, score in zip(sim_ids, sim_scores):
                    r = df_valid[df_valid["product_id"] == pid_sim]
                    if not r.empty:
                        r = r.iloc[0]
                        rows.append(
                            {
                                "mode": mode_name,
                                "neighbor_product_id": pid_sim,
                                "similarity_score": score,
                                "PROD_COD": r.get("PROD_COD"),
                                "PROD_DES": r.get("PROD_DES"),
                                "CLR_DES": r.get("CLR_DES"),
                            }
                        )
                return pd.DataFrame(rows)

            with tab_text:
                st.markdown("**Top-4 neighbours (Text only)**")
                df_text = build_neighbor_table(rec_dfs["Text only"], "Text only")
                if df_text.empty:
                    st.info("No text-only neighbours found for this product.")
                else:
                    st.dataframe(df_text)

            with tab_image:
                st.markdown("**Top-4 neighbours (Image only – CLIP)**")
                df_img = build_neighbor_table(rec_dfs["Image only (CLIP)"], "Image only (CLIP)")
                if df_img.empty:
                    st.info("No CLIP neighbours found for this product.")
                else:
                    st.dataframe(df_img)

            with tab_hybrid:
                st.markdown("**Top-4 neighbours (Hybrid – Text + Image)**")
                df_hyb = build_neighbor_table(rec_dfs["Hybrid (Text + Image)"], "Hybrid (Text + Image)")
                if df_hyb.empty:
                    st.info("No hybrid neighbours found for this product.")
                else:
                    st.dataframe(df_hyb)

            # Optional: combined comparison in an expander
            if not (df_text.empty and df_img.empty and df_hyb.empty):
                with st.expander("Compare neighbours across all modes", expanded=False):
                    combined = pd.concat(
                        [df for df in [df_text, df_img, df_hyb] if not df.empty],
                        ignore_index=True,
                    )
                    st.dataframe(combined)
# -------------------------------------------------
# 4. Experiment with the text–image weight α
# -------------------------------------------------
st.subheader("4. Experiment with the text–image weight α")

if T_emb is None or I_emb is None or not hybrid_ids or df_products.empty:
    st.info(
        "Interactive weight tuning is not available. "
        "Make sure 'text_embeddings.npy', 'clip_image_embeddings.npy', "
        "'product_ids_text.csv' and 'product_ids_image.csv' exist in "
        "the 'similarity_pipeline' folder."
    )
else:
    df_alpha = df_products.copy()
    if "product_id" not in df_alpha.columns:
        if "PROD_CLR" in df_alpha.columns:
            df_alpha["product_id"] = df_alpha["PROD_CLR"]
        else:
            st.error(
                "df_product.csv has no 'product_id' or 'PROD_CLR' column. "
                "Cannot align products with embeddings."
            )
            st.stop()

    df_alpha = df_alpha[df_alpha["product_id"].isin(hybrid_ids)].copy()
    if df_alpha.empty:
        st.warning("No overlap between products and embedding product_ids.")
    else:
        def make_label2(row):
            return (
                f"{row['PROD_COD']} | {row['CLR_DES']} | "
                f"{row['product_id']} | {str(row['PROD_DES'])[:50]}"
            )

        df_alpha["label"] = df_alpha.apply(make_label2, axis=1)
        df_alpha = df_alpha.sort_values("label")
        label_to_pid2 = dict(zip(df_alpha["label"], df_alpha["product_id"]))

        # --- Top row: selector + product image + formula ---
        col_left, col_center, col_right = st.columns([2, 1.2, 2])

        with col_left:
            selected_label2 = st.selectbox(
                "Select a product to tune α:",
                options=list(label_to_pid2.keys()),
            )
            selected_pid2 = label_to_pid2[selected_label2]

            alpha = st.slider(
                "Weight for TEXT similarity α",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Hybrid similarity = α·Text + (1−α)·Image",
            )

        with col_center:
            paths_sel2 = image_map.get(selected_pid2, [])
            st.markdown("**Selected product image**")
            if paths_sel2:
                st.image(paths_sel2[0], caption=selected_pid2, use_container_width=True)
            else:
                st.info("No image found for this product.")

        with col_right:
            st.markdown(
                f"""
                **Current hybrid formula:**  
                \\( S_{{hybrid}} = {alpha:.2f} \\cdot S_{{text}} + {1-alpha:.2f} \\cdot S_{{image}} \\)
                """
            )

        # Compact title right under the top row (small margins)
        st.markdown(
            """
            <div style="
                font-size:24px;
                font-weight:600;
                margin-top:0.05rem;
                margin-bottom:0.3rem;">
                Neighbours for the selected product with current α
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Compute similarities for this product only ---
        idx_pid = hybrid_ids.index(selected_pid2)
        t_vec = T_emb[idx_pid]          # (d_text,)
        i_vec = I_emb[idx_pid]          # (d_img,)

        s_text = t_vec @ T_emb.T        # (N,)
        s_img  = i_vec @ I_emb.T        # (N,)
        s_hyb  = alpha * s_text + (1 - alpha) * s_img

        # Exclude self
        s_text[idx_pid] = -1.0
        s_img[idx_pid]  = -1.0
        s_hyb[idx_pid]  = -1.0

        # Top-k neighbours
        K = 4
        idx_sorted = np.argsort(-s_hyb)[:K]

        rows_alpha = []
        for j in idx_sorted:
            pid_j = hybrid_ids[j]
            r = df_alpha[df_alpha["product_id"] == pid_j]
            if not r.empty:
                r = r.iloc[0]
                rows_alpha.append(
                    {
                        "neighbor_product_id": pid_j,
                        "text_score": float(s_text[j]),
                        "image_score": float(s_img[j]),
                        "hybrid_score": float(s_hyb[j]),
                        "PROD_COD": r.get("PROD_COD"),
                        "PROD_DES": r.get("PROD_DES"),
                        "CLR_DES": r.get("CLR_DES"),
                    }
                )

        if rows_alpha:
            df_neighbors_alpha = pd.DataFrame(rows_alpha)

            # --- IMAGES FIRST (closer to the table) ---
            st.markdown(
                """
                <div style="
                    font-size:18px;
                    font-weight:500;
                    margin-top:0.05rem;   /* was 0.4rem */
                    margin-bottom:0.1rem;">
                    Visual view of neighbours (with current α)
                </div>
                """,
                unsafe_allow_html=True,
            )

            cols = st.columns(len(df_neighbors_alpha))

            for col, (_, row_n) in zip(cols, df_neighbors_alpha.iterrows()):
                pid_n = row_n["neighbor_product_id"]
                paths_n = image_map.get(pid_n, [])

                with col:
                    st.markdown(f"**{pid_n}**")
                    st.caption(
                        f"Hybrid: {row_n['hybrid_score']:.3f}  \n"
                        f"Text: {row_n['text_score']:.3f} | "
                        f"Image: {row_n['image_score']:.3f}"
                    )

                    # small negative margin to pull image up
                    st.markdown(
                        '<div style="margin-top:-0.4rem;"></div>',
                        unsafe_allow_html=True,
                    )

                    if paths_n:
                        # smaller image (adjust width if you want)
                        st.image(paths_n[0], width=220)
                    else:
                        st.info("No image available")


            # --- TABLE AFTER IMAGES ---
            st.markdown(
                """
                <div style="
                    font-size:18px;
                    font-weight:500;
                    margin-top:0.6rem;
                    margin-bottom:0.2rem;">
                    Tabular view of neighbours (scores with current α)
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(df_neighbors_alpha)
        else:
            st.info("No neighbours found for this configuration.")




# -------------------------------------------------
# 5. References
# -------------------------------------------------
st.subheader("References to go further on the subject")

st.markdown(
    """
- Radford, A., Kim, J. W., Hallacy, C., et al. (2021).  
  *Learning transferable visual models from natural language supervision (CLIP).*

- Reimers, N., & Gurevych, I. (2019).  
  *Sentence-BERT: Sentence embeddings using Siamese BERT-Networks.*

- Aggarwal, C. C. (2016).  
  *Recommender Systems: The Textbook.* Springer.

- Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019).  
  *Multimodal machine learning: A survey and taxonomy.*
"""
)
