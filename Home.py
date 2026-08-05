import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Penerapan Model BERT untuk Named Entity Recognition dan Analisis Topik dengan LDA pada Listing Produk Skincare",
    page_icon="🧴",
    layout="wide"
)

st.markdown("""
<style>

/* Memperbesar tulisan menu sidebar */
section[data-testid="stSidebar"] ul li div p{
    font-size: 20px !important;
    font-weight: bold;
}

/* Memperbesar judul sidebar */
section[data-testid="stSidebar"]{
    font-size:20px;
}

</style>
""", unsafe_allow_html=True)
# =====================================
# LOAD DATASET
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_final.csv")

df = load_data()

# =====================================
# HEADER
# =====================================

st.title("🧴 Penerapan Model BERT untuk Named Entity Recognition dan Analisis Topik dengan LDA pada Listing Produk Skincare")

st.markdown(
    """
    <p style="font-size:18px; color:gray; margin-top:-10px;">
        Dibuat Oleh: <b>Irma Octavia Chaniago</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.image(
    "assets/banner.jpeg",
    use_container_width=True
)

st.markdown("""

Dashboard ini merupakan implementasi penelitian yang bertujuan untuk mengekstraksi informasi penting dari data produk skincare menggunakan model IndoBERT serta melakukan analisis topik menggunakan metode Latent Dirichlet Allocation (LDA). Melalui kombinasi kedua metode tersebut, sistem mampu mengidentifikasi entitas penting berupa merek, jenis produk, kandungan, dan manfaat, serta menemukan topik-topik dominan pada data produk skincare.
""")

# =====================================
# INFORMASI DATASET
# =====================================

st.markdown("---")

st.subheader("📊 Informasi Dataset")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Jumlah Data",
        f"{len(df):,}"
    )

with c2:
    st.markdown("**Kategori**")
    st.markdown(
        "<h2 style='margin-top:0;'>Perawatan dan Kecantikan</h2>",
        unsafe_allow_html=True
    )

with c3:
    st.markdown("**Sumber Data**")
    st.markdown(
        "<h2 style='margin-top:0;'>Marketplace Shopee</h2>",
        unsafe_allow_html=True
    )

# =====================================
# LATAR BELAKANG
# =====================================

st.markdown("---")

st.subheader("📖 Latar Belakang")

st.write("""
Pesatnya perkembangan industri perawatan dan kecantikan di Indonesia menyebabkan jumlah produk yang tersedia pada marketplace terus meningkat. Informasi penting seperti nama merek, jenis produk, kandungan aktif, dan manfaat produk umumnya masih tersimpan dalam bentuk teks tidak terstruktur pada nama produk.

Kondisi tersebut menyebabkan proses analisis data menjadi kurang efektif apabila dilakukan secara manual. Oleh karena itu, penelitian ini memanfaatkan pendekatan Natural Language Processing (NLP) melalui Named Entity Recognition (NER) berbasis IndoBERT untuk mengekstraksi informasi penting dari nama produk skincare, serta metode Latent Dirichlet Allocation (LDA) untuk mengidentifikasi topik-topik utama yang terdapat pada kumpulan data produk.
""")

# =====================================
# TUJUAN PENELITIAN
# =====================================

st.markdown("---")

st.subheader("🎯 Tujuan Penelitian")

st.markdown("""
- Mengekstraksi entitas penting dari data produk skincare menggunakan model IndoBERT.
- Mengidentifikasi entitas berupa merek, jenis produk, kandungan, dan manfaat.
- Melakukan analisis topik menggunakan metode Latent Dirichlet Allocation (LDA).
- Menampilkan hasil analisis dalam bentuk dashboard interaktif berbasis Streamlit.
""")

# =====================================
# METODE
# =====================================

st.markdown("---")

st.subheader("🧠 Metode yang Digunakan")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
### IndoBERT

Model bahasa berbasis BERT yang digunakan untuk proses Named Entity Recognition (NER).
""")

with col2:
    st.info("""
### Named Entity Recognition

Digunakan untuk mendeteksi entitas:

- MEREK
- JENIS_PRODUK
- KANDUNGAN
- MANFAAT
""")

with col3:
    st.info("""
### Latent Dirichlet Allocation

Digunakan untuk mengidentifikasi topik-topik dominan pada data produk skincare.
""")

# =====================================
# ENTITAS
# =====================================

st.markdown("---")

st.subheader("🏷️ Entitas yang Diekstraksi")

c1, c2, c3, c4 = st.columns(4)

c1.success("🏷️ MEREK")
c2.success("🧴 JENIS_PRODUK")
c3.success("🧪 KANDUNGAN")
c4.success("✨ MANFAAT")

# =====================================
# HALAMAN DASHBOARD
# =====================================

st.markdown("---")

st.subheader("📌 Halaman Dashboard")

st.info("""
📊 Dashboard Analisa

Menampilkan karakteristik dataset melalui distribusi merek, jenis produk, kandungan, manfaat, dan distribusi topik.

🧠 Analisis

Memungkinkan pengguna memasukkan deskripsi produk untuk dilakukan ekstraksi entitas menggunakan model IndoBERT serta identifikasi distribusi probabilitas topik menggunakan model LDA.
""")