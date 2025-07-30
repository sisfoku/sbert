import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# -------------------------------
# 1. Konfigurasi Halaman
# -------------------------------
st.set_page_config(page_title="🎓 Rekomendasi Kurikulum OBE", layout="wide")
st.title("🎓 Sistem Rekomendasi Kurikulum OBE")
st.markdown("Menggunakan **SBERT Fine-Tuned** untuk pencocokan profil lulusan")

# -------------------------------
# 2. Cek File yang Dibutuhkan
# -------------------------------
required_files = [
    'fine_tuned_sbert_obe',
    'kurikulum_obe_structured.json'
]

missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    st.error(f"File tidak ditemukan: {', '.join(missing)}")
    st.stop()

# -------------------------------
# 3. Muat Model SBERT yang Sudah Di-Fine-Tune
# -------------------------------
@st.cache_resource
def load_model():
    st.info("📥 Memuat model SBERT yang sudah di-fine-tune...")
    try:
        model = SentenceTransformer('fine_tuned_sbert_obe')
        st.success("✅ Model SBERT berhasil dimuat.")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

model = load_model()

# -------------------------------
# 4. Muat Data Kurikulum
# -------------------------------
@st.cache_data
def load_curriculum_data():
    try:
        with open('kurikulum_obe_structured.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        st.success(f"✅ Data dari {len(data)} profil lulusan berhasil dimuat.")
        return data
    except Exception as e:
        st.error(f"❌ Gagal memuat data kurikulum: {e}")
        st.stop()

curriculum_data = load_curriculum_data()

# -------------------------------
# 5. Encode Semua Profil Lulusan
# -------------------------------
@st.cache_data
def encode_profiles(data):
    st.info("🧠 Melakukan encoding pada semua profil lulusan...")
    corpus = []
    corpus_data = []
    for item in 
        text = f"{item['program_studi']}: {item['profil_lulusan']}"
        corpus.append(text)
        corpus_data.append(item)
    embeddings = model.encode(corpus)
    st.success(f"✅ {len(corpus)} profil telah di-encode.")
    return corpus, corpus_data, embeddings

corpus, corpus_data, corpus_embeddings = encode_profiles(curriculum_data)

# -------------------------------
# 6. Fungsi Rekomendasi
# -------------------------------
def recommend_curriculum(prodi, profil):
    query = f"{prodi}: {profil}"
    query_emb = model.encode([query])
    sims = cosine_similarity(query_emb, corpus_embeddings)[0]
    best_idx = sims.argmax()
    best_item = corpus_data[best_idx]
    score = sims[best_idx]
    return {
        "similarity_score": float(score),
        "matched_prodi": best_item["program_studi"],
        "matched_profil": best_item["profil_lulusan"],
        "cpl_list": best_item["cpl_list"]
    }

# -------------------------------
# 7. Antarmuka Pengguna
# -------------------------------
st.markdown("---")
st.subheader("🔍 Masukkan Input")

col1, col2 = st.columns([1, 3])
with col1:
    program_studi = st.selectbox(
        "Program Studi",
        ["Sistem Informasi", "Sistem Komputer"]
    )
with col2:
    profil_lulusan = st.text_input(
        "Profil Lulusan",
        placeholder="Contoh: lulusan mampu melakukan analisis data untuk pengambilan keputusan"
    )

if st.button("🚀 Dapatkan Rekomendasi"):
    if not profil_lulusan.strip():
        st.error("Mohon masukkan deskripsi profil lulusan.")
    else:
        with st.spinner("Mencari profil terdekat..."):
            result = recommend_curriculum(program_studi, profil_lulusan)

        # Tampilkan hasil
        st.markdown("---")
        st.subheader("🎯 Hasil Rekomendasi")
        st.markdown(f"**Program Studi Input:** `{program_studi}`")
        st.markdown(f"**Profil Input:** `{profil_lulusan}`")
        st.markdown(f"**Profil Terdekat:** `{result['matched_profil']}`")
        st.markdown(f"**Similarity Score:** `{result['similarity_score']:.3f}`")

        st.markdown("---")
        st.subheader("📋 CPL yang Direkomendasikan")
        for cpl in result['cpl_list']:
            st.markdown(f"  • **{cpl['kode']}**: {cpl['deskripsi']}")
            for cpmk in cpl['cpmk_list']:
                st.markdown(f"    → {cpmk['kode']}: {cpmk['deskripsi']}")
                st.markdown(f"       {cpmk['kode_mk']} - {cpmk['nama_mk']}")

# -------------------------------
# 8. Info di Sidebar
# -------------------------------
with st.sidebar:
    st.header("ℹ️ Informasi")
    st.markdown("""
    - **Model**: SBERT (fine-tuned dengan triplet loss)
    - **Data**: kurikulum_obe_structured.json
    - **Metode**: Cosine Similarity
    - Cocok untuk sistem pendukung pengembangan kurikulum OBE.
    """)
