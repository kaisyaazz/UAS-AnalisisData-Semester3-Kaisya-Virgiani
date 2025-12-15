import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(
    page_title="Dashboard Ketimpangan Fasilitas Kesehatan",
    layout="wide"
)

st.title("Dashboard Ketimpangan Fasilitas Kesehatan Indonesia")

st.markdown("""
Dashboard ini menampilkan hasil **clustering fasilitas kesehatan**
untuk mengidentifikasi dan mengevaluasi ketimpangan layanan kesehatan
antar provinsi di Indonesia.
""")

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("fitur.csv")

# st.write("Kolom CSV:", df.columns.tolist())

df.columns = df.columns.str.lower().str.replace(" ", "_")

# =====================
# METRIC CARDS
# =====================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Jumlah Provinsi", df["provinsi_faskes"].nunique())
col2.metric("Jumlah Cluster", df["cluster"].nunique())
col3.metric("Total Faskes", int(df["jumlah_faskes"].sum()))
col4.metric("Total Kunjungan", int(df["jumlah_kunjungan"].sum()))

st.divider()
#====================
#Penjelasan Clustering
#====================
st.subheader("Karakteristik Cluster Fasilitas Kesehatan")

cluster_desc = {
    0: {
        "nama": "Wilayah Tertinggal",
        "deskripsi": (
            "Cluster ini ditandai dengan jumlah fasilitas kesehatan "
            "dan kunjungan layanan yang relatif rendah. "
            "Wilayah dalam cluster ini berpotensi mengalami keterbatasan akses "
            "dan memerlukan prioritas intervensi kebijakan."
        )
    },
    1: {
        "nama": "Wilayah Tertekan",
        "deskripsi": (
            "Cluster ini memiliki tingkat kunjungan layanan yang cukup tinggi, "
            "namun jumlah dan kapasitas fasilitas kesehatan masih terbatas. "
            "Kondisi ini mengindikasikan beban layanan yang tidak seimbang."
        )
    },
    2: {
        "nama": "Wilayah Relatif Maju",
        "deskripsi": (
            "Cluster ini dicirikan oleh ketersediaan fasilitas kesehatan "
            "dan bobot layanan yang relatif tinggi. "
            "Wilayah dalam cluster ini memiliki akses dan kapasitas layanan "
            "yang lebih memadai dibandingkan cluster lainnya."
        )
    }
}

for k, v in cluster_desc.items():
    st.markdown(f"""
**Cluster {k} – {v['nama']}**  
{v['deskripsi']}
""")

# =====================
# GRAFIK 1: DISTRIBUSI CLUSTER
# =====================
st.subheader("Distribusi Provinsi per Cluster")
st.bar_chart(df["cluster"].value_counts().sort_index())

# =====================
# GRAFIK 2: FASKES PER CLUSTER
# =====================
st.subheader("Rata-rata Jumlah Fasilitas Kesehatan per Cluster")
st.bar_chart(df.groupby("cluster")["jumlah_faskes"].mean())

# =====================
# GRAFIK 3: KUNJUNGAN PER CLUSTER
# =====================
st.subheader("Rata-rata Kunjungan Layanan per Cluster")
st.bar_chart(df.groupby("cluster")["jumlah_kunjungan"].mean())

# =====================
# GRAFIK 4: BOBOT FASKES
# =====================
st.subheader("Rata-rata Bobot Fasilitas per Cluster")
st.bar_chart(df.groupby("cluster")["rata_bobot"].mean())

st.divider()

# =====================
# MESIN ANALISIS CLUSTER
# =====================
st.subheader("Analisis Cluster Wilayah Baru")

with st.form("cluster_form"):
    jf = st.number_input(
        "Jumlah Fasilitas Kesehatan",
        min_value=0.0,
        step=1.0,
        key="jf"
    )
    jk = st.number_input(
        "Jumlah Kunjungan",
        min_value=0.0,
        step=1.0,
        key="jk"
    )
    rb = st.number_input(
        "Rata-rata Bobot Fasilitas",
        min_value=0.0,
        step=0.1,
        key="rb"
    )
    tb = st.number_input(
        "Total Bobot Fasilitas",
        min_value=0.0,
        step=1.0,
        key="tb"
    )

    submit = st.form_submit_button("Analisis")

if submit:
    model = pickle.load(open("model_clustering.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))

    X = np.array([[jf, jk, rb, tb]])
    X_scaled = scaler.transform(X)
    cluster = model.predict(X_scaled)[0]

    st.success(f"Wilayah ini termasuk dalam **Cluster {cluster}**")

    if cluster == 0:
        st.info("Cluster dengan keterbatasan fasilitas dan prioritas tinggi intervensi kebijakan.")
    elif cluster == 1:
        st.info("Cluster dengan fasilitas menengah dan potensi pengembangan.")
    else:
        st.info("Cluster dengan fasilitas relatif memadai.")

    st.subheader("Rekomendasi Kebijakan Kesehatan")

    if cluster == 0:
        st.warning("⚠️ Prioritas Intervensi Tinggi")
        st.markdown("""
        **Karakteristik Wilayah:**
        - Jumlah fasilitas kesehatan terbatas
        - Tingkat kunjungan relatif rendah
        - Akses layanan kesehatan masih kurang memadai

        **Rekomendasi Kebijakan:**
        - Pembangunan fasilitas kesehatan baru (puskesmas/klinik dasar)
        - Pemerataan tenaga kesehatan (dokter, perawat, bidan)
        - Peningkatan anggaran kesehatan daerah
        - Program layanan kesehatan keliling untuk wilayah terpencil
        """)

    elif cluster == 1:
        st.info("⚖️ Optimalisasi & Penguatan Layanan")
        st.markdown("""
        **Karakteristik Wilayah:**
        - Jumlah kunjungan tinggi
        - Fasilitas kesehatan belum sebanding dengan kebutuhan
        - Beban layanan cenderung tinggi

        **Rekomendasi Kebijakan:**
        - Peningkatan kapasitas fasilitas kesehatan yang ada
        - Penambahan jam layanan dan tenaga medis
        - Optimalisasi sistem rujukan antar fasilitas
        - Digitalisasi layanan (pendaftaran, antrian, rekam medis)
        """)

    elif cluster == 2:
        st.success("✅ Pemeliharaan & Replikasi Praktik Baik")
        st.markdown("""
        **Karakteristik Wilayah:**
        - Fasilitas kesehatan relatif memadai
        - Bobot dan kualitas layanan tinggi
        - Akses layanan kesehatan baik

        **Rekomendasi Kebijakan:**
        - Pemeliharaan kualitas fasilitas dan layanan
        - Pengembangan layanan kesehatan spesialistik
        - Replikasi praktik terbaik ke wilayah cluster lain
        - Monitoring dan evaluasi berkelanjutan
        """)

