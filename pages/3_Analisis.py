import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)

from gensim.models import LdaModel
from gensim.corpora import Dictionary


# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="Analisis Query",
    page_icon="🧠",
    layout="wide"
)

# Model dari Hugging Face
MODEL_PATH = "irmaoctavia/indobert-ner-skincare"


# ==================================================
# LOAD NER
# ==================================================

@st.cache_resource
def load_ner():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_PATH
    )

    ner = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1
    )

    return ner


# ==================================================
# LOAD LDA
# ==================================================

@st.cache_resource
def load_lda():

    lda_model = LdaModel.load(
        "lda_model.model_indobert"
    )

    dictionary = Dictionary.load(
        "dictionary.dict_indobert"
    )

    return lda_model, dictionary


ner_model = load_ner()

lda_model, dictionary = load_lda()


# ==================================================
# NAMA TOPIK
# ==================================================

TOPIC_NAMES = {
    0: "Skincare Anti-Aging dan Brightening",
    1: "Pembersih Wajah dan Perawatan Jerawat",
    2: "Pelembap, Retinol, dan Perlindungan Kulit",
    3: "Hidrasi dan Perawatan Kulit Sensitif",
    4: "Brightening dengan Bahan Aktif"
}


# ==================================================
# HEADER
# ==================================================

st.title("🧠 Analisis Query")

query = st.text_area(
    "Masukkan deskripsi produk",
    height=150
)


# ==================================================
# ANALISIS
# ==================================================

if st.button("🔍 Analisis"):

    if query.strip() == "":

        st.warning(
            "Masukkan teks terlebih dahulu"
        )

    else:

        # ==========================================
        # NER
        # ==========================================

        hasil = ner_model(query)

        # ------------------------------------------
        # Bersihkan token ##
        # ------------------------------------------

        def clean_entity(text):

            text = text.replace(" ##", "")
            text = text.replace("##", "")

            text = (
                text
                .replace(" ,", ",")
                .replace(" .", "")
            )

            text = " ".join(text.split())

            return text.strip()


        # ==========================================
        # HASIL NER
        # ==========================================

        rows = []

        i = 0

        while i < len(hasil):

            current = hasil[i]

            entity = current["word"].replace("##", "")
            label = current["entity_group"]

            start = current["start"]
            end = current["end"]

            j = i + 1

            # Gabungkan subword yang berurutan
            while (
                j < len(hasil)
                and hasil[j]["entity_group"] == label
                and hasil[j]["word"].startswith("##")
                and hasil[j]["start"] == end
            ):

                entity += hasil[j]["word"].replace("##", "")
                end = hasil[j]["end"]
                j += 1

            rows.append({
                "Entitas": entity,
                "Label": label
            })

            i = j

        pred_df = pd.DataFrame(rows)
        # ==========================================
        # POST-PROCESSING HASIL NER
        # ==========================================

        import re

        STOP_ENTITY = {
            "dan",
            "atau",
            "dengan",
            "dgn",
            "dngan",
            "utk",
            "untk",
            "buat",
            "untuk",
            "serta",
            "hingga",
            "sampai",
            "di",
            "ke",
            "dari",
            "pada",
            "yang",
            "yg",
            "&"
        }

        def clean_entity_result(text):

            text = str(text).strip()

            # hapus token subword
            text = text.replace("##", "")

            # ganti simbol menjadi spasi
            text = re.sub(r"[&/+|]", " ", text)

            # hapus kata hubung di AWAL entitas
            text = re.sub(
                r'^(dan|atau|dengan|dgn|dngan|utk|untk|buat|untuk|serta|hingga|sampai)\s+',
                '',
                text,
                flags=re.IGNORECASE
            )

            # hapus kata hubung di AKHIR entitas
            text = re.sub(
                r'\s+(dan|atau|dengan|dgn|dngan|utk|untk|buat|untuk|serta|hingga|sampai)$',
                '',
                text,
                flags=re.IGNORECASE
            )

            # hapus simbol di awal/akhir
            text = re.sub(r'^[^\w]+|[^\w]+$', '', text)

            # rapikan spasi
            text = " ".join(text.split())

            return text

        # bersihkan entitas
        pred_df["Entitas"] = pred_df["Entitas"].apply(clean_entity_result)

        # hapus entitas kosong
        pred_df = pred_df[
            pred_df["Entitas"].str.strip() != ""
        ]

        # hapus jika hanya stopword
        pred_df = pred_df[
            ~pred_df["Entitas"]
                .str.lower()
                .isin(STOP_ENTITY)
        ]

        pred_df = pred_df.reset_index(drop=True)

        st.subheader("🏷 Hasil Named Entity Recognition")

        st.dataframe(
            pred_df,
            use_container_width=True,
            hide_index=True
        )

        # ==========================================
        # Fungsi Ambil Entitas
        # ==========================================

        def ambil(label):

            data = pred_df[
                pred_df["Label"] == label
            ]

            if data.empty:
                return "-"

            entities = (
                data["Entitas"]
                .astype(str)
                .tolist()
            )

            hasil = ""

            for ent in entities:

                if ent.startswith("##"):
                    hasil += ent.replace("##", "")

                else:

                    if hasil != "":
                        hasil += ", "

                    hasil += ent

            return hasil


        # ==========================================
        # LDA
        # ==========================================

        tokens = query.lower().split()

        bow = dictionary.doc2bow(tokens)

        topics = lda_model.get_document_topics(
            bow,
            minimum_probability=0
        )

        topic_df = pd.DataFrame(
            [
                {
                    "Topik": TOPIC_NAMES[topic_id],
                    "Probabilitas (%)": round(prob * 100, 2)
                }

                for topic_id, prob in topics
            ]
        )

        topic_df = topic_df.sort_values(
            by="Probabilitas (%)",
            ascending=False
        )

        # ==========================================
        # TOPIK DOMINAN
        # ==========================================

        top_topic = topic_df.iloc[0]

        st.success(
            f"""
Topik Dominan

{top_topic['Topik']}

Probabilitas :

{top_topic['Probabilitas (%)']}%
"""
        )


        # ==========================================
        # KEYWORD TOPIK
        # ==========================================

        topic_id = max(
            topics,
            key=lambda x: x[1]
        )[0]

        keywords = lda_model.show_topic(
            topic_id,
            topn=10
        )

        keyword_text = ", ".join(
            [
                word
                for word, _
                in keywords
            ]
        )

        st.subheader(
            "🔑 Keyword Topik Dominan"
        )

        st.info(
            keyword_text
        )