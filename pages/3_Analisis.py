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

        for h in hasil:

            entitas = clean_entity(h["word"])

            label = h["entity_group"]

            if entitas in [".", ",", ";", ":", "!", "?"]:
                continue

            rows.append({

                "Entitas": entitas,

                "Label": label

            })

        pred_df = pd.DataFrame(rows)

        st.subheader(
            "🏷 Hasil Named Entity Recognition"
        )

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