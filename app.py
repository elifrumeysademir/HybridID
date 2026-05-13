"""
HybridID - S7: Streamlit Kullanıcı Arayüzü
============================================
Görevler:
  - S7/P0: Streamlit kullanıcı paneli geliştirme
  - S7/P0: Arayüz ve analiz modüllerinin bağlanması
  - S7/P1: PDF/CSV çıktı sisteminin kurulması

Çalıştırma:
  streamlit run app.py
"""

import os
import sys
import tempfile
import time

import streamlit as st
from PIL import Image

# ── sys.path: src/ modüllerini bul ──────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_ROOT, "src")
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hybrid_score    import load_cnn_model, run_full_analysis
from xai_engine      import save_gradcam
from report_generator import generate_pdf_report, generate_csv_report

# ─────────────────────────────────────────────────────────────────────────────
# Sayfa yapılandırması
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid-ID | Görsel Doğrulama Sistemi",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Özel CSS — Dark theme & bileşenler
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Genel arkaplan ─────────────────────────────── */
.stApp { background: #0f0f1a; color: #e2e8f0; }

/* ── Sidebar ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid #2d2d4e;
}
[data-testid="stSidebar"] * { color: #c7d2fe !important; }

/* ── Metric kartları ────────────────────────────── */
[data-testid="stMetric"] {
    background: #1e1e38;
    border: 1px solid #3730a3;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricLabel"] { color: #818cf8 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.6rem !important; }

/* ── Progress bar ───────────────────────────────── */
.stProgress > div > div { border-radius: 10px; }

/* ── Butonlar ───────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white; border: none; border-radius: 10px;
    padding: 10px 24px; font-weight: 600; font-size: 1rem;
    transition: all 0.3s ease; width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
}

/* ── Kart container'ları ────────────────────────── */
.verdict-card {
    border-radius: 16px; padding: 28px 32px;
    text-align: center; margin-bottom: 20px;
    animation: fadeIn 0.6s ease;
}
.verdict-fake {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 2px solid #ef4444;
    box-shadow: 0 0 40px rgba(239,68,68,0.3);
}
.verdict-real {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 2px solid #22c55e;
    box-shadow: 0 0 40px rgba(34,197,94,0.3);
}
.verdict-title { font-size: 2.4rem; font-weight: 800; margin: 0; letter-spacing: 0.05em; }
.verdict-fake .verdict-title { color: #fca5a5; }
.verdict-real .verdict-title { color: #86efac; }
.verdict-sub { font-size: 1rem; opacity: 0.8; margin-top: 8px; }

/* ── Section başlıkları ─────────────────────────── */
.section-header {
    font-size: 1.15rem; font-weight: 700; color: #818cf8;
    border-bottom: 2px solid #3730a3; padding-bottom: 6px;
    margin-bottom: 14px; margin-top: 8px;
}

/* ── Skor satırı ────────────────────────────────── */
.score-row {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 10px; padding: 10px 14px;
    background: #1e1e38; border-radius: 10px;
    border: 1px solid #2d2d4e;
}
.score-label { min-width: 140px; font-size: 0.85rem; color: #94a3b8; }
.score-value { min-width: 50px; font-weight: 700; color: #e2e8f0; font-size: 1.05rem; }

/* ── Info kutusu ────────────────────────────────── */
.info-box {
    background: #1e1e38; border: 1px solid #2d2d4e;
    border-radius: 12px; padding: 16px; margin-bottom: 14px;
}

/* ── Animasyon ──────────────────────────────────── */
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

/* ── Tab stilleri ───────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { background: #1a1a2e; border-radius: 10px; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #94a3b8; border-radius: 8px; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #4f46e5 !important; color: white !important; }

/* ── File uploader ──────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #1e1e38; border: 2px dashed #3730a3;
    border-radius: 14px; padding: 20px;
}

/* ── Tablo ──────────────────────────────────────── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CNN Modeli — Uygulama ömrü boyunca bir kez yüklenir
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_model():
    return load_cnn_model()


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────
def _confidence_badge(confidence: str) -> str:
    colors = {"YÜKSEK": "#22c55e", "ORTA": "#f59e0b", "DÜŞÜK": "#ef4444"}
    c = colors.get(confidence, "#6b7280")
    return f'<span style="background:{c};color:white;padding:3px 10px;border-radius:20px;font-size:0.8rem;font-weight:700">{confidence}</span>'


def _score_bar(label: str, score: float, weight: float):
    """Katman skoru için renkli progress bar çizer."""
    bar_color = "#ef4444" if score > 0.5 else "#22c55e"
    st.markdown(f"""
    <div class="score-row">
        <div class="score-label">⚡ {label}</div>
        <div class="score-value">{score:.2f}</div>
        <div style="flex:1">
            <div style="background:#2d2d4e;border-radius:8px;height:10px;overflow:hidden">
                <div style="width:{score*100:.0f}%;height:100%;background:{bar_color};
                            border-radius:8px;transition:width 1s ease"></div>
            </div>
        </div>
        <div style="min-width:50px;color:#6366f1;font-size:0.8rem">w={weight:.0%}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px">
        <div style="font-size:2.5rem">🔍</div>
        <div style="font-size:1.3rem;font-weight:800;color:#818cf8;margin-top:4px">Hybrid-ID</div>
        <div style="font-size:0.75rem;color:#6366f1;margin-top:2px">Görsel Doğrulama Sistemi</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📋 Sistem Bilgisi")
    st.markdown("""
    <div class="info-box">
    <b>Analiz Katmanları:</b><br>
    🔵 Metadata (EXIF) — %20<br>
    🟡 ELA Analizi — %35<br>
    🔴 CNN (ResNet50) — %45
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Ayarlar")
    run_gradcam = st.toggle("Grad-CAM hesapla", value=True,
                             help="Kapatmak analizi hızlandırır ama XAI görselini devre dışı bırakır.")
    save_reports = st.toggle("Analiz çıktısını kaydet", value=True,
                              help="Isı haritaları ve Grad-CAM görselleri outputs/ klasörüne kaydedilir.")

    st.divider()
    st.markdown('<div style="font-size:0.7rem;color:#4b5563;text-align:center">v1.0 · Sprint 7</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Ana içerik
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="color:#818cf8;font-size:2rem;font-weight:800;margin-bottom:4px">
    🔍 Hybrid-ID — Görsel Sahtelik Dedektörü
</h1>
<p style="color:#6b7280;font-size:0.95rem;margin-bottom:24px">
    EXIF Metadata + ELA Görsel Analizi + ResNet50 CNN ile çok katmanlı görsel doğrulama.
</p>
""", unsafe_allow_html=True)

# ── Görsel Yükleme ─────────────────────────────────────────────────────────
col_upload, col_preview = st.columns([1.2, 1])

with col_upload:
    st.markdown('<div class="section-header">📁 Görsel Yükle</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "JPG, JPEG veya PNG yükleyin",
        type=["jpg", "jpeg", "png"],
        key="main_uploader",
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.success(f"✅ **{uploaded_file.name}** yüklendi ({uploaded_file.size / 1024:.1f} KB)")
        analyze_btn = st.button("🚀 Analizi Başlat", key="analyze_btn", use_container_width=True)
    else:
        st.info("⬆️ Bir görsel yükleyerek analizi başlatın.")
        analyze_btn = False

with col_preview:
    if uploaded_file:
        st.markdown('<div class="section-header">👁️ Önizleme</div>', unsafe_allow_html=True)
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, use_container_width=True, caption=f"{pil_img.size[0]}×{pil_img.size[1]} px")


# ─────────────────────────────────────────────────────────────────────────────
# Analiz Akışı
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file and analyze_btn:
    # Geçici dosyaya yaz
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Model yükle
    with st.spinner("🔄 CNN modeli hazırlanıyor..."):
        model = _get_model()

    # Analiz progress bar
    progress_bar = st.progress(0, text="Analiz başlatılıyor...")
    status_text  = st.empty()

    t_start = time.time()

    try:
        status_text.markdown("**[1/4]** Metadata analizi yapılıyor...")
        progress_bar.progress(15, text="Metadata analizi...")

        # Tam analiz
        result = run_full_analysis(tmp_path, model=model)

        progress_bar.progress(60, text="ELA & CNN tamamlandı...")
        status_text.markdown("**[3/4]** Grad-CAM hesaplanıyor...")

        # Grad-CAM (isteğe bağlı)
        gradcam_path = None
        if run_gradcam and model:
            try:
                gradcam_path = save_gradcam(tmp_path, model)
                result["gradcam_path"] = gradcam_path
            except Exception as e:
                st.warning(f"Grad-CAM oluşturulamadı: {e}")

        progress_bar.progress(90, text="Rapor hazırlanıyor...")
        status_text.markdown("**[4/4]** Sonuçlar derleniyor...")

        # PDF & CSV üret
        pdf_bytes = generate_pdf_report(
            result,
            ela_heatmap_path=result.get("ela_heatmap_path"),
            gradcam_path=gradcam_path,
        )
        csv_str = generate_csv_report(result)

        elapsed = time.time() - t_start
        progress_bar.progress(100, text=f"Tamamlandı! ({elapsed:.1f}sn)")
        status_text.empty()

        # Geçici dosyayı koru (Streamlit session boyunca okunacak)
        st.session_state["result"]      = result
        st.session_state["pdf_bytes"]   = pdf_bytes
        st.session_state["csv_str"]     = csv_str
        st.session_state["tmp_path"]    = tmp_path
        st.session_state["gradcam_path"]= gradcam_path
        st.session_state["analyzed"]    = True

    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ Analiz hatası: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        st.session_state["analyzed"] = False


# ─────────────────────────────────────────────────────────────────────────────
# Sonuç Paneli
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("analyzed") and "result" in st.session_state:
    result      = st.session_state["result"]
    pdf_bytes   = st.session_state["pdf_bytes"]
    csv_str     = st.session_state["csv_str"]
    gradcam_path= st.session_state.get("gradcam_path")

    verdict      = result["verdict"]
    hybrid_score = result["hybrid_score"]
    confidence   = result["confidence"]
    layer_scores = result["layer_scores"]
    weights      = result["weights"]

    st.markdown("---")

    # ── Verdict Kartı ─────────────────────────────────────────────────────────
    verdict_class = "verdict-fake" if verdict == "FAKE" else "verdict-real"
    verdict_icon  = "🔴" if verdict == "FAKE" else "🟢"

    if verdict == "FAKE":
        verdict_tr = result.get("manipulation_type", "MANİPÜLE / SAHTE")
        if "YAPAY ZEKA" in verdict_tr:
            verdict_icon = "🤖"
        elif "MONTAJ" in verdict_tr:
            verdict_icon = "✂️"
    else:
        verdict_tr = "GERÇEK / TEMİZ"

    badge_html    = _confidence_badge(confidence)

    st.markdown(f"""
    <div class="verdict-card {verdict_class}">
        <div class="verdict-title">{verdict_icon} {verdict_tr}</div>
        <div class="verdict-sub">
            Sahtelik (Manipülasyon) İhtimali: <b>{hybrid_score*100:.1f}%</b>
            &nbsp;&nbsp;{badge_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Özet metrikler ────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Sahtelik İhtimali",    f"% {hybrid_score*100:.1f}")
    m2.metric("🧠 CNN Skoru",      f"{layer_scores.get('cnn', 0):.4f}")
    m3.metric("🔬 ELA Skoru",      f"{layer_scores.get('ela', 0):.4f}")
    m4.metric("🏷️ Meta Skoru",     f"{layer_scores.get('metadata', 0):.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sekmeli detaylar ──────────────────────────────────────────────────────
    tab_scores, tab_visuals, tab_report = st.tabs([
        "📊 Katman Skorları", "🔥 Görsel Analizler", "📋 Rapor & İndirme"
    ])

    # ── Tab 1: Katman Skorları ────────────────────────────────────────────────
    with tab_scores:
        st.markdown('<div class="section-header">Ağırlıklı Katman Skorları</div>',
                    unsafe_allow_html=True)

        _score_bar("Metadata (EXIF)",  layer_scores.get("metadata", 0), weights.get("metadata", 0))
        _score_bar("ELA Analizi",       layer_scores.get("ela", 0),      weights.get("ela", 0))
        _score_bar("CNN (ResNet50)",    layer_scores.get("cnn", 0),      weights.get("cnn", 0))

        st.markdown("<br>", unsafe_allow_html=True)

        # Hibrit skor formülü
        st.markdown(f"""
        <div class="info-box">
        <b>🔢 Hibrit Skor Formülü:</b><br>
        <code>
        HybridScore = {weights['metadata']:.0%}×{layer_scores['metadata']:.3f} (Meta)
                    + {weights['ela']:.0%}×{layer_scores['ela']:.3f} (ELA)
                    + {weights['cnn']:.0%}×{layer_scores['cnn']:.3f} (CNN)
                    = <b>{hybrid_score:.4f}</b>
        </code>
        </div>
        """, unsafe_allow_html=True)

        # Şüpheli bölgeler
        regions = result.get("ela_result", {}).get("suspicious_regions", [])
        if regions:
            st.markdown('<div class="section-header">📍 ELA — Şüpheli Bölgeler</div>',
                        unsafe_allow_html=True)
            import pandas as pd
            df = pd.DataFrame(regions)
            df.index = df.index + 1
            df.columns = ["X", "Y", "Genişlik", "Yükseklik", "Skor"]
            st.dataframe(df, use_container_width=True)

        # Metadata bilgisi
        meta_res = result.get("metadata_result", {})
        if meta_res.get("is_ai_detected"):
            sigs = meta_res.get("ai_signatures_found", [])
            st.error(f"🚨 **AI İmzası Tespit Edildi!** ({len(sigs)} imza)")
            for s in sigs[:5]:
                st.markdown(f"  - {s}")
        elif meta_res.get("success"):
            st.success("✅ Metadata temiz — Bilinen AI aracı izi bulunamadı.")
        else:
            st.info(f"ℹ️ Metadata: {meta_res.get('error_message', 'Analiz yapılamadı')}")

    # ── Tab 2: Görsel Analizler ───────────────────────────────────────────────
    with tab_visuals:
        ela_path = result.get("ela_heatmap_path")
        gc_path  = gradcam_path or result.get("gradcam_path")

        if ela_path and os.path.exists(ela_path):
            st.markdown('<div class="section-header">🔬 ELA — Hata Seviyesi Analizi</div>',
                        unsafe_allow_html=True)
            st.image(ela_path, use_container_width=True,
                     caption="Sol: Orijinal | Orta: ELA Isı Haritası | Sağ: Tespit Edilen Bölgeler")
        else:
            st.info("ELA ısı haritası mevcut değil.")

        st.markdown("<br>", unsafe_allow_html=True)

        if gc_path and os.path.exists(gc_path):
            st.markdown('<div class="section-header">🔥 Grad-CAM — CNN Aktivasyon Haritası (XAI)</div>',
                        unsafe_allow_html=True)
            st.image(gc_path, use_container_width=True,
                     caption="Sol: Orijinal | Orta: Grad-CAM Haritası | Sağ: Süperpozisyon")
        else:
            st.info("Grad-CAM görselleştirmesi mevcut değil.")

    # ── Tab 3: Rapor & İndirme ────────────────────────────────────────────────
    with tab_report:
        st.markdown('<div class="section-header">📥 Raporu İndir</div>', unsafe_allow_html=True)

        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            st.download_button(
                label="📄 PDF Raporu İndir",
                data=pdf_bytes,
                file_name=f"hybridid_rapor_{os.path.splitext(uploaded_file.name)[0]}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="pdf_download",
            )
            st.markdown('<div style="text-align:center;color:#6b7280;font-size:0.8rem">Tam rapor: görseller + tablolar</div>',
                        unsafe_allow_html=True)

        with dl_col2:
            st.download_button(
                label="📊 CSV Raporu İndir",
                data=csv_str,
                file_name=f"hybridid_rapor_{os.path.splitext(uploaded_file.name)[0]}.csv",
                mime="text/csv",
                use_container_width=True,
                key="csv_download",
            )
            st.markdown('<div style="text-align:center;color:#6b7280;font-size:0.8rem">Sayısal veriler: skorlar + koordinatlar</div>',
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # EXIF tablosu
        exif = result.get("metadata_result", {}).get("extracted_metadata", {})
        if exif:
            st.markdown('<div class="section-header">🏷️ EXIF Metadata Detayları</div>',
                        unsafe_allow_html=True)
            import pandas as pd
            exif_df = pd.DataFrame(list(exif.items()), columns=["Etiket", "Değer"])
            st.dataframe(exif_df, use_container_width=True, height=280)
        else:
            st.info("EXIF metadata bulunamadı veya görsel temizlenmiş.")

# Touch for Streamlit reload
