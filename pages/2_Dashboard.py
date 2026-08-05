import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(
    page_title="Dashboard Analisis Produk Skincare",
    page_icon="🧴",
    layout="wide"
)

# ==========================================================
# LOAD DATA
# ==========================================================
DATA_PATH = "data/dataset_final.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # antisipasi missing value
    # hanya isi NA untuk entitas
    cols = [
        "merek",
        "jenis_produk",
        "kandungan",
        "manfaat"
    ]

    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna("Tidak diketahui")

    return df


df = load_data()


# ==========================================================
# MEMBERSIHKAN DATA UNTUK VISUALISASI
# ==========================================================
def data_valid(series):

    return (
        series
        .dropna()
        .astype(str)
        .loc[
            ~series.astype(str).str.lower().isin(
                [
                    "tidak diketahui",
                    "nan",
                    ""
                ]
            )
        ]
    )

# ==========================================================
# FUNGSI WORDCLOUD
# ==========================================================
def tampil_wordcloud(series):

    stop_words = {
        "tidak",
        "diketahui",
        "untuk",
        "dan",
        "yang",
        "di",
        "dengan",
        "atau",
        "&",
        "utk",
        "untk",
        "dgn",
        "dngan",
        "dngn",
        "pkai",
        "pakai",
        "guna",

    }

    text = " ".join(
        data_valid(series)
    )

    if text.strip() == "":
        st.warning("Data tidak tersedia")
        return

    wc = WordCloud(
        width=1200,
        height=500,
        background_color="white",
        stopwords=stop_words
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12,4))

    ax.imshow(wc)

    ax.axis("off")

    st.pyplot(fig)


# ==========================================================
# HEADER
# ==========================================================
st.title("🧴 Dashboard Analisis Produk Skincare")

st.markdown(
"""
Dashboard ini menampilkan distribusi entitas dan karakteristik topik
hasil analisis dataset produk skincare.
"""
)

# ==========================================================
# OVERVIEW
# ==========================================================
st.header("📌 Ringkasan Dataset")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Jumlah Data",
    len(df)
)

c2.metric(
    "Jumlah Merek",
    df["merek"].nunique()
)

c3.metric(
    "Jumlah Jenis Produk",
    df["jenis_produk"].nunique()
)

c4, c5, c6 = st.columns(3)

c4.metric(
    "Jumlah Kandungan",
    df["kandungan"].nunique()
)

c5.metric(
    "Jumlah Manfaat",
    df["manfaat"].nunique()
)

c6.metric(
    "Jumlah Topik",
    df["topik"].nunique()
)

# ==========================================================
# DISTRIBUSI MEREK
# ==========================================================
st.header("🏷 Distribusi Merek")

top_merek = (
    data_valid(df["merek"])
    .value_counts()
    .head(15)
)

fig, ax = plt.subplots(figsize=(8,5))

ax.barh(
    top_merek.index[::-1],
    top_merek.values[::-1]
)

ax.set_xlabel("Frekuensi")

st.pyplot(fig)

# ==========================================================
# DISTRIBUSI JENIS PRODUK
# ==========================================================
st.header("🧴 Distribusi Jenis Produk")

top_jenis = (
    data_valid(df["jenis_produk"])
    .value_counts()
    .head(15)
)

fig, ax = plt.subplots(figsize=(10,6))

ax.barh(
    top_jenis.index[::-1],
    top_jenis.values[::-1]
)

ax.set_xlabel("Frekuensi")
ax.set_ylabel("Jenis Produk")

plt.tight_layout()

st.pyplot(fig)

# ==========================================================
# DISTRIBUSI KANDUNGAN
# ==========================================================
st.header("🧪 Distribusi Kandungan")

top_kandungan = (
    data_valid(df["kandungan"])
    .value_counts()
    .head(15)
)

fig, ax = plt.subplots(figsize=(8,5))

ax.barh(
    top_kandungan.index[::-1],
    top_kandungan.values[::-1]
)

st.pyplot(fig)

# ==========================================================
# DISTRIBUSI MANFAAT
# ==========================================================
st.header("✨ Distribusi Manfaat")

top_manfaat = (
    data_valid(df["manfaat"])
    .value_counts()
    .head(15)
)

fig, ax = plt.subplots(figsize=(8,5))

ax.barh(
    top_manfaat.index[::-1],
    top_manfaat.values[::-1]
)

st.pyplot(fig)

# ==========================================================
# DISTRIBUSI TOPIK
# ==========================================================
st.header("🧠 Distribusi Topik")

topik_dist = (
    df["topik_label"]
    .value_counts()
)

fig, ax = plt.subplots(figsize=(9,5))

bars = ax.bar(
    topik_dist.index,
    topik_dist.values
)

for bar in bars:

    tinggi = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width()/2,
        tinggi,
        str(int(tinggi)),
        ha='center'
    )

plt.xticks(rotation=15)

st.pyplot(fig)

# ==========================================================
# DATASET
# ==========================================================
st.header("📄 Dataset")

st.dataframe(
    df,
    use_container_width=True
)