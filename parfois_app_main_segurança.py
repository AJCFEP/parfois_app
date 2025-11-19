import streamlit as st
import pandas as pd
import os


import streamlit as st
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)
LOGO_PATH = os.path.join(BASE_DIR, "parfois.png")  # or os.path.join(BASE_DIR, "images", "parfois.png")



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

######


import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="PARFOIS – Similarity", layout="wide")

# Reduce top padding
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.8rem;   /* default is ~5rem; try 0–1 */
        }
    </style>
    """,
    unsafe_allow_html=True,
)




######

IMAGES_DIR = os.path.join(BASE_DIR, "images")

@st.cache_data
def build_image_map(images_root: str = IMAGES_DIR):
    """
    Scan the images folder and build a mapping:
    product_id -> list of image paths

    Assumes filenames like 140486_BM_1.jpg
    so product_id = '140486_BM'.
    """
    mapping = {}
    for root, dirs, files in os.walk(images_root):
        for fname in files:
            fname_lower = fname.lower()
            if fname_lower.endswith((".jpg", ".jpeg", ".png")):
                parts = fname.split("_")
                if len(parts) >= 2:
                    product_id = "_".join(parts[:2])  # '140486_BM_1.jpg' -> '140486_BM'
                else:
                    product_id = os.path.splitext(fname)[0]

                full_path = os.path.join(root, fname)
                mapping.setdefault(product_id, []).append(full_path)

    return mapping

image_map = build_image_map()



######


df_products, df_sales, rec_df = load_data()

# --- UI ---

# --- Logo + subtitle stacked on the left ---


# --- Header: logo on the left, subtitle on the right ---
col_logo, col_title = st.columns([2, 3])  # adjust ratio if needed

with col_logo:
    # Fill the column; column width will control the logo size
    st.image(LOGO_PATH, use_container_width=True)

with col_title:
    # Subtitle aligned roughly mid-height beside the logo
    st.markdown(
        """
        <div style="
            font-family:Arial;
            font-size:26px;
            color:#555;
            margin-top:2.5rem;   /* move up/down to align with logo */
            margin-bottom:0.2rem;
        ">
            Similarity Detection for Fashion Retail Products
        </div>
        """,
        unsafe_allow_html=True,
    )

# Thin horizontal line below both
st.markdown("<hr>", unsafe_allow_html=True)


st.markdown("---")


#st.write("✅ Streamlit app is running.")
#st.write(f"- **Products loaded:** {len(df_products)} rows")
#st.write(f"- **Sales loaded:** {len(df_sales)} rows")
#st.write(f"- **Recommendations loaded:** {len(rec_df)} rows")

#st.write("Here are the first 5 products:")
#st.dataframe(df_products.head())


########################3

# Global style adjustments
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    h2 { margin-top: 0.2rem !important; }
</style>
""", unsafe_allow_html=True)

#st.markdown("---")
st.header("Explore product similarities")

# --- Prepare df_products with a 'product_id' column that matches rec_df ---
if "product_id" not in df_products.columns:
    # In your notebook you used PROD_CLR as product_id
    df_products = df_products.copy()
    df_products["product_id"] = df_products["PROD_CLR"]

# Keep only products that appear in rec_df (so we know we have recommendations)
valid_ids = set(rec_df["product_id"])
df_prod_valid = df_products[df_products["product_id"].isin(valid_ids)].copy()

# --- Build options for the scroll list ---
def make_label(row):
    return f"{row['PROD_COD']} | {row['CLR_DES']} | {row['product_id']} | {row['PROD_DES'][:50]}"

df_prod_valid["label"] = df_prod_valid.apply(make_label, axis=1)

# Sort alphabetically by label
df_prod_valid = df_prod_valid.sort_values("label")

label_to_pid = dict(zip(df_prod_valid["label"], df_prod_valid["product_id"]))

selected_label = st.selectbox(
    "Choose a product:",
    options=list(label_to_pid.keys()),
    index=0
)

selected_pid = label_to_pid[selected_label]

# --- Show selected product info ---
st.subheader("Selected product")

row_sel = df_prod_valid[df_prod_valid["product_id"] == selected_pid].iloc[0]

st.write({
    "product_id": selected_pid,
    "PROD_COD": row_sel["PROD_COD"],
    "PROD_DES": row_sel["PROD_DES"],
    "CLR_DES": row_sel["CLR_DES"],
})

# Show image of the selected product
paths_sel = image_map.get(selected_pid, [])
if paths_sel:
    st.image(paths_sel[0], caption=f"{selected_pid}", width=300)
else:
    st.info("No image found for this product in the images folder.")

# --- Show 4 most similar products ---
st.subheader("Top 4 similar products")

rec_row = rec_df[rec_df["product_id"] == selected_pid]
if rec_row.empty:
    st.warning("No recommendations found for this product.")
else:
    rec_row = rec_row.iloc[0]
    sim_ids = [
        rec_row["sim_1_id"],
        rec_row["sim_2_id"],
        rec_row["sim_3_id"],
        rec_row["sim_4_id"],
    ]
    sim_scores = [
        rec_row["sim_1_score"],
        rec_row["sim_2_score"],
        rec_row["sim_3_score"],
        rec_row["sim_4_score"],
    ]

    cols = st.columns(4)
    rows_info = []

    for col, pid_sim, score in zip(cols, sim_ids, sim_scores):
        with col:
            col.markdown(f"**{pid_sim}**")
            col.caption(f"Similarity: {score:.3f}")

            # Info from df_prod_valid if available
            r = df_prod_valid[df_prod_valid["product_id"] == pid_sim]
            if not r.empty:
                r = r.iloc[0]
                col.write(r["PROD_DES"])
                col.caption(r["CLR_DES"])

            # Image
            paths = image_map.get(pid_sim, [])
            if paths:
                col.image(paths[0], use_container_width=True)
            else:
                col.info("No image")

            # For the table at the bottom
            info_row = {
                "product_id": pid_sim,
                "similarity": score,
                "PROD_COD": r["PROD_COD"] if not r.empty else None,
                "PROD_DES": r["PROD_DES"] if not r.empty else None,
                "CLR_DES": r["CLR_DES"] if not r.empty else None,
            }
            rows_info.append(info_row)

    # Table with attributes of the 4 similar products
    if rows_info:
        st.markdown("#### Details of similar products")
        st.dataframe(pd.DataFrame(rows_info))
