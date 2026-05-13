"""
HybridID - S7/P1: Rapor Üreticisi (PDF + CSV)
===============================================
Görev:
  Hibrit analiz sonuçlarını PDF ve CSV formatında çıktı olarak üretir.

Kullanım (import):
  from report_generator import generate_pdf_report, generate_csv_report

Fonksiyonlar:
  generate_pdf_report(result, ela_path, gradcam_path) → bytes
  generate_csv_report(result)                          → str
"""

import os
import io
import csv
import sys
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# ReportLab imports
# ─────────────────────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Image as RLImage,
    KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF

# ─────────────────────────────────────────────────────────────────────────────
# Renk paleti (dark theme uyumlu baskı renkleri)
# ─────────────────────────────────────────────────────────────────────────────
C_DARK      = colors.HexColor("#1a1a2e")
C_PRIMARY   = colors.HexColor("#4f46e5")   # indigo
C_FAKE_RED  = colors.HexColor("#ef4444")
C_REAL_GRN  = colors.HexColor("#22c55e")
C_ACCENT    = colors.HexColor("#6366f1")
C_LIGHT_BG  = colors.HexColor("#f0f4ff")
C_GRAY      = colors.HexColor("#6b7280")
C_TEXT      = colors.HexColor("#1e293b")
C_WHITE     = colors.white
C_BORDER    = colors.HexColor("#e2e8f0")


# ─────────────────────────────────────────────────────────────────────────────
# Stil şablonları
# ─────────────────────────────────────────────────────────────────────────────
def _build_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["title"] = ParagraphStyle(
        "title",
        parent=base["Title"],
        fontSize=22,
        fontName="Helvetica-Bold",
        textColor=C_DARK,
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle",
        parent=base["Normal"],
        fontSize=10,
        fontName="Helvetica",
        textColor=C_GRAY,
        spaceAfter=16,
        alignment=TA_CENTER,
    )
    styles["section"] = ParagraphStyle(
        "section",
        parent=base["Heading2"],
        fontSize=13,
        fontName="Helvetica-Bold",
        textColor=C_PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
        borderPad=0,
    )
    styles["body"] = ParagraphStyle(
        "body",
        parent=base["Normal"],
        fontSize=9,
        fontName="Helvetica",
        textColor=C_TEXT,
        spaceAfter=4,
    )
    styles["small"] = ParagraphStyle(
        "small",
        parent=base["Normal"],
        fontSize=7.5,
        fontName="Helvetica",
        textColor=C_GRAY,
    )
    styles["verdict_fake"] = ParagraphStyle(
        "verdict_fake",
        parent=base["Normal"],
        fontSize=28,
        fontName="Helvetica-Bold",
        textColor=C_FAKE_RED,
        alignment=TA_CENTER,
    )
    styles["verdict_real"] = ParagraphStyle(
        "verdict_real",
        parent=base["Normal"],
        fontSize=28,
        fontName="Helvetica-Bold",
        textColor=C_REAL_GRN,
        alignment=TA_CENTER,
    )
    return styles


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı: Görsel boyutlandırma
# ─────────────────────────────────────────────────────────────────────────────
def _fit_image(path: str, max_w_cm: float, max_h_cm: float) -> RLImage | None:
    """Görseli PDF sayfasına sığacak şekilde orantılı boyutlandırır."""
    if not path or not os.path.exists(path):
        return None
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as img:
            orig_w, orig_h = img.size
        max_w = max_w_cm * cm
        max_h = max_h_cm * cm
        scale = min(max_w / orig_w, max_h / orig_h, 1.0)
        return RLImage(path, width=orig_w * scale, height=orig_h * scale)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı: Skor çubuğu çizimi
# ─────────────────────────────────────────────────────────────────────────────
def _score_bar_table(layer_scores: dict, weights: dict) -> Table:
    """Her katman için etiket + skor + progress bar içeren tablo döner."""
    bar_max_w = 8 * cm
    row_h     = 0.55 * cm

    data = [["Katman", "Ağırlık", "Skor", "Görsel"]]
    labels = {"metadata": "Metadata (EXIF)", "ela": "ELA Analizi", "cnn": "CNN Modeli"}

    for key in ("metadata", "ela", "cnn"):
        score  = layer_scores.get(key, 0.0)
        weight = weights.get(key, 0.0)

        # Çubuk çizimi
        d = Drawing(bar_max_w, row_h)
        # Arka plan
        d.add(Rect(0, 2, bar_max_w, row_h - 4, fillColor=C_BORDER, strokeColor=None))
        # Dolum
        fill_color = C_FAKE_RED if score > 0.5 else C_REAL_GRN
        fill_w     = max(bar_max_w * score, 2)
        d.add(Rect(0, 2, fill_w, row_h - 4, fillColor=fill_color, strokeColor=None))

        data.append([
            labels[key],
            f"{weight:.0%}",
            f"{score:.2f}",
            d,
        ])

    t = Table(data, colWidths=[4 * cm, 2 * cm, 1.8 * cm, bar_max_w + 0.4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), C_PRIMARY),
        ("TEXTCOLOR",   (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE",    (0, 1), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT_BG]),
        ("GRID",        (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Ana Fonksiyon: PDF Üretimi
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_report(
    result: dict,
    ela_heatmap_path: str | None = None,
    gradcam_path: str | None = None,
) -> bytes:
    """
    Hibrit analiz sonuçlarından PDF raporu üretir.

    Args:
        result          : `hybrid_score.run_full_analysis()` çıktısı.
        ela_heatmap_path: ELA ısı haritası PNG yolu (None ise atlanır).
        gradcam_path    : Grad-CAM PNG yolu (None ise atlanır).

    Returns:
        PDF içeriği bytes olarak (Streamlit download_button'a doğrudan verilebilir).
    """
    buffer  = io.BytesIO()
    doc     = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
    )
    styles  = _build_styles()
    content = []

    verdict      = result.get("verdict", "?")
    hybrid_score = result.get("hybrid_score", 0.0)
    confidence   = result.get("confidence", "?")
    layer_scores = result.get("layer_scores", {})
    weights      = result.get("weights", {})
    file_path    = result.get("file_path", "Bilinmiyor")
    timestamp    = datetime.now().strftime("%d.%m.%Y %H:%M")

    # ── 1. Başlık ─────────────────────────────────────────────────────────────
    content.append(Paragraph("Hybrid-ID Analiz Raporu", styles["title"]))
    content.append(Paragraph(
        f"Dosya: <b>{os.path.basename(file_path)}</b> &nbsp;|&nbsp; Tarih: {timestamp}",
        styles["subtitle"]
    ))
    content.append(HRFlowable(width="100%", thickness=1.5, color=C_PRIMARY, spaceAfter=10))

    # ── 2. Karar Kutusu ────────────────────────────────────────────────────────
    verdict_color  = C_FAKE_RED if verdict == "FAKE" else C_REAL_GRN
    verdict_label  = "🔴 MANİPÜLE / SAHTE" if verdict == "FAKE" else "🟢 GERÇEK / TEMİZ"
    score_pct      = f"{hybrid_score * 100:.1f}%"

    verdict_data = [[
        Paragraph(verdict_label, ParagraphStyle(
            "vl", fontName="Helvetica-Bold", fontSize=16,
            textColor=verdict_color, alignment=TA_CENTER
        )),
        Paragraph(
            f"<b>Hybrid Skor:</b> {score_pct}<br/>"
            f"<b>Güven:</b> {confidence}",
            ParagraphStyle("vs", fontName="Helvetica", fontSize=11,
                           textColor=C_TEXT, alignment=TA_CENTER)
        ),
    ]]
    verdict_table = Table(verdict_data, colWidths=[9 * cm, 8.5 * cm])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_LIGHT_BG),
        ("BOX",           (0, 0), (-1, -1), 2, verdict_color),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    content.append(KeepTogether([verdict_table, Spacer(1, 10)]))

    # ── 3. Katman Skoru Tablosu ───────────────────────────────────────────────
    content.append(Paragraph("Katman Bazlı Analiz", styles["section"]))
    content.append(_score_bar_table(layer_scores, weights))
    content.append(Spacer(1, 8))

    # ── 4. ELA Isı Haritası ───────────────────────────────────────────────────
    ela_path = ela_heatmap_path or result.get("ela_heatmap_path")
    ela_img  = _fit_image(ela_path, max_w_cm=16, max_h_cm=7)
    if ela_img:
        content.append(Paragraph("ELA Analizi — Hata Seviyesi Haritası", styles["section"]))
        content.append(ela_img)
        content.append(Spacer(1, 6))

    # ── 5. Grad-CAM ────────────────────────────────────────────────────────────
    gc_path = gradcam_path or result.get("gradcam_path")
    gc_img  = _fit_image(gc_path, max_w_cm=16, max_h_cm=7)
    if gc_img:
        content.append(Paragraph("Grad-CAM — CNN Aktivasyon Haritası (XAI)", styles["section"]))
        content.append(gc_img)
        content.append(Spacer(1, 6))

    # ── 6. Şüpheli Bölgeler ────────────────────────────────────────────────────
    regions = result.get("ela_result", {}).get("suspicious_regions", [])
    if regions:
        content.append(Paragraph("ELA — Şüpheli Bölge Koordinatları", styles["section"]))
        reg_data = [["#", "X", "Y", "Genişlik", "Yükseklik", "Bölge Skoru"]]
        for i, r in enumerate(regions[:15], 1):   # en fazla 15 bölge
            reg_data.append([str(i), r["x"], r["y"], r["w"], r["h"], f"{r['score']:.4f}"])
        reg_table = Table(reg_data, colWidths=[1*cm, 2.2*cm, 2.2*cm, 2.5*cm, 2.5*cm, 3*cm])
        reg_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), C_ACCENT),
            ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LIGHT_BG]),
            ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        content.append(reg_table)
        content.append(Spacer(1, 8))

    # ── 7. Metadata (EXIF) Tablosu ────────────────────────────────────────────
    exif_data = result.get("metadata_result", {}).get("extracted_metadata", {})
    if exif_data:
        content.append(Paragraph("Metadata — EXIF Verileri", styles["section"]))
        exif_table_data = [["Etiket", "Değer"]]
        shown = 0
        for k, v in exif_data.items():
            val_str = str(v)[:80] + ("…" if len(str(v)) > 80 else "")
            exif_table_data.append([k, val_str])
            shown += 1
            if shown >= 30:  # En fazla 30 EXIF satırı
                break
        exif_table = Table(exif_table_data, colWidths=[7 * cm, 10.5 * cm])
        exif_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), C_ACCENT),
            ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 7.5),
            ("ALIGN",         (0, 0), (0, -1), "LEFT"),
            ("ALIGN",         (1, 0), (1, -1), "LEFT"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LIGHT_BG]),
            ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ]))
        content.append(exif_table)

    # ── 8. Footer ─────────────────────────────────────────────────────────────
    content.append(Spacer(1, 14))
    content.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
    content.append(Paragraph(
        "Bu rapor Hybrid-ID sistemi tarafından otomatik olarak üretilmiştir. "
        "Sonuçlar referans amaçlıdır; kesin karar için uzman değerlendirmesi önerilir.",
        styles["small"]
    ))

    doc.build(content)
    return buffer.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# CSV Çıktısı
# ─────────────────────────────────────────────────────────────────────────────
def generate_csv_report(result: dict) -> str:
    """
    Hibrit analiz sonuçlarını CSV formatında döner.

    Returns:
        CSV içeriği UTF-8 string olarak.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    timestamp = datetime.now().strftime("%d.%m.%Y %H:%M")

    # Özet
    writer.writerow(["Hybrid-ID Analiz Özeti"])
    writer.writerow(["Dosya",          os.path.basename(result.get("file_path", ""))])
    writer.writerow(["Tarih",          timestamp])
    writer.writerow(["Karar",          result.get("verdict", "?")])
    writer.writerow(["Hybrid Skor",    result.get("hybrid_score", "")])
    writer.writerow(["Güven",          result.get("confidence", "")])
    writer.writerow([])

    # Katman skorları
    writer.writerow(["Katman", "Ağırlık", "Skor"])
    ls = result.get("layer_scores", {})
    ws = result.get("weights", {})
    for key, label in [("metadata", "Metadata EXIF"), ("ela", "ELA Analizi"), ("cnn", "CNN Modeli")]:
        writer.writerow([label, ws.get(key, ""), ls.get(key, "")])
    writer.writerow([])

    # Şüpheli bölgeler
    regions = result.get("ela_result", {}).get("suspicious_regions", [])
    if regions:
        writer.writerow(["Şüpheli Bölgeler"])
        writer.writerow(["#", "X", "Y", "Genişlik", "Yükseklik", "Skor"])
        for i, r in enumerate(regions, 1):
            writer.writerow([i, r["x"], r["y"], r["w"], r["h"], r["score"]])
        writer.writerow([])

    # EXIF metadata
    exif = result.get("metadata_result", {}).get("extracted_metadata", {})
    if exif:
        writer.writerow(["EXIF Metadata"])
        writer.writerow(["Etiket", "Değer"])
        for k, v in exif.items():
            writer.writerow([k, str(v)[:200]])

    return output.getvalue()
