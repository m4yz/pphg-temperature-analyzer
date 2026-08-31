import re
import streamlit as st
# FINAL REPORT LAYOUT REVISION: compact management tables; interpretation notes below tables.
import pandas as pd
import plotly.express as px
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.charts.legends import Legend

st.set_page_config(page_title="PPHG Temperature Analyzer", page_icon="🌡️", layout="wide")

st.title("🌡️ PPHG Temperature Analyzer")
st.caption("Upload Testo CSV → automatic PPHG temperature analysis")

# PPHG SOP
RULES = {
    "Chiller": {"limit": 6.0, "delay": pd.Timedelta(hours=2)},
    "Freezer": {"limit": -15.0, "delay": pd.Timedelta(hours=4)},
}

# Equipment category is detected from the equipment code in its name.
# The uploaded CSV determines which equipment exists; no fixed equipment list
# is required. Display Category is specific, while PPHG Group drives the
# Chiller/Freezer threshold rule.
EQUIPMENT_CODE_CATEGORY = {
    "UPC": ("Upright Chiller", "Chiller"),
    "UPCS": ("Upright Chiller Showcase", "Chiller"),
    "UDC": ("Under Counter Chiller", "Chiller"),
    "UCC": ("Under Counter Chiller", "Chiller"),
    "WCH": ("Wine Chiller", "Chiller"),
    "SCH": ("Showcase Chiller", "Chiller"),
    "SHC": ("Showcase Chiller", "Chiller"),
    "CHS": ("Showcase Chiller", "Chiller"),
    "UPF": ("Upright Freezer", "Freezer"),
    "UCF": ("Under Counter Freezer", "Freezer"),
    "BTC": ("Bottle Chiller", "Chiller"),
}

def category_from_name(name):
    """Resolve display category + PPHG group from an equipment code."""
    name = str(name).strip()
    for code, (display_category, pphg_group) in EQUIPMENT_CODE_CATEGORY.items():
        if re.search(rf"(^|[._\-\s]){re.escape(code)}([._\-\s]|$)", name, re.IGNORECASE):
            return display_category, pphg_group

    # Compatibility fallback for older exports without the standardized code.
    low = name.lower()
    if "freezer" in low:
        return "Freezer", "Freezer"
    if "chiller" in low:
        return "Chiller", "Chiller"

    return "Other", "Other"

def format_duration(td):
    if pd.isna(td):
        return "—"
    total_minutes = max(0, round(td.total_seconds() / 60))
    days, rem = divmod(total_minutes, 1440)
    hours, minutes = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)

def longest_continuous(g, category, median_interval):
    if category not in RULES:
        return pd.Timedelta(0), pd.NaT, pd.NaT, pd.NaT, "N/A"

    rule = RULES[category]
    g = g.sort_values("Timestamp").copy()
    g["above"] = g["Temperature"] >= rule["limit"]

    # A missing/gapped sample must not be treated as continuous threshold event.
    max_gap = median_interval * 1.5 if median_interval > pd.Timedelta(0) else pd.Timedelta(minutes=10)
    g["gap"] = g["Timestamp"].diff() > max_gap
    g["group"] = (g["gap"] | ~g["above"]).cumsum()

    runs = g[g["above"]].groupby("group", sort=False)
    best = None
    for _, r in runs:
        if r.empty:
            continue
        start = r["Timestamp"].iloc[0]
        end = r["Timestamp"].iloc[-1]
        duration = end - start
        if best is None or duration > best[0]:
            best = (duration, start, end, r["Temperature"].max())

    if best is None:
        return pd.Timedelta(0), pd.NaT, pd.NaT, pd.NaT, "NORMAL"

    duration, start, end, peak = best
    status = "ALARM" if duration >= rule["delay"] else "WARNING"
    return duration, start, end, peak, status


def excursion_stats(g, category, median_interval):
    """Calculate distinct threshold events and the longest continuous event.

    A threshold event is one distinct period in which the measured temperature is
    at or above the applicable PPHG limit. This is a data-derived recurrence
    indicator only; it is NOT a count of equipment failures or root-cause events.
    """
    if category not in RULES:
        return {
            "duration": pd.Timedelta(0), "start": pd.NaT, "end": pd.NaT,
            "peak": pd.NaT, "status": "N/A", "threshold_events": 0,
        }

    rule = RULES[category]
    g = g.sort_values("Timestamp").copy()
    g["above"] = g["Temperature"] >= rule["limit"]

    # A sampling gap must not be treated as continuous time above threshold.
    max_gap = (
        median_interval * 1.5
        if median_interval > pd.Timedelta(0)
        else pd.Timedelta(minutes=10)
    )
    g["gap"] = g["Timestamp"].diff() > max_gap
    g["group"] = (g["gap"] | ~g["above"]).cumsum()

    runs = []
    for _, r in g[g["above"]].groupby("group", sort=False):
        if r.empty:
            continue
        runs.append({
            "start": r["Timestamp"].iloc[0],
            "end": r["Timestamp"].iloc[-1],
            "duration": r["Timestamp"].iloc[-1] - r["Timestamp"].iloc[0],
            "peak": r["Temperature"].max(),
        })

    if not runs:
        return {
            "duration": pd.Timedelta(0), "start": pd.NaT, "end": pd.NaT,
            "peak": pd.NaT, "status": "NORMAL", "threshold_events": 0,
        }

    best = max(runs, key=lambda x: x["duration"])

    if best["duration"] <= pd.Timedelta(0):
        status = "SINGLE POINT"
    elif best["duration"] >= rule["delay"]:
        status = "ALARM"
    else:
        status = "WARNING"

    return {
        "duration": best["duration"],
        "start": best["start"],
        "end": best["end"],
        "peak": best["peak"],
        "status": status,
        "threshold_events": len(runs),
    }


def parse_testo(uploaded_files):
    """Parse one or more Testo CSV exports and combine all measurements."""
    if not uploaded_files:
        raise ValueError("Belum ada CSV Testo yang dipilih.")

    records = []
    raw_frames = []
    file_names = []

    for uploaded in uploaded_files:
        raw = pd.read_csv(uploaded, sep=";", encoding="utf-8-sig")
        if raw.shape[1] < 2:
            continue

        raw_frames.append(raw)
        file_names.append(getattr(uploaded, "name", "Testo CSV"))

        timestamp_col = raw.columns[0]
        timestamps = pd.to_datetime(raw[timestamp_col], errors="coerce")
        valid_time = timestamps.notna()

        # Analyze every measurement column actually present in this file.
        for col in raw.columns[1:]:
            values = pd.to_numeric(raw[col], errors="coerce")
            mask = valid_time & values.notna()
            if mask.sum() == 0:
                continue

            equipment_name = str(col).split(": Temperature")[0].strip()
            category, pphg_group = category_from_name(equipment_name)

            records.append(pd.DataFrame({
                "Timestamp": timestamps[mask].values,
                "Equipment": equipment_name,
                "Temperature": values[mask].values,
                "Category": category,
                "PPHG Group": pphg_group,
                "Source File": getattr(uploaded, "name", "Testo CSV"),
            }))

    if not records:
        raise ValueError("Tidak ada measurement temperature yang dapat dibaca dari CSV.")

    data = pd.concat(records, ignore_index=True)
    data = data.sort_values(["Equipment", "Timestamp"]).reset_index(drop=True)

    # Keep raw data available for compatibility with the existing report flow.
    raw = pd.concat(raw_frames, ignore_index=True, sort=False) if raw_frames else pd.DataFrame()
    return data, raw, file_names

def analyze(df):
    rows = []
    intervals = []

    for _, g in df.groupby("Equipment", sort=False):
        diffs = g["Timestamp"].sort_values().diff().dropna()
        diffs = diffs[diffs > pd.Timedelta(0)]
        intervals.extend(diffs.tolist())

    median_interval = (
        pd.Series(intervals).median()
        if intervals else pd.Timedelta(minutes=5)
    )

    for equipment, g in df.groupby("Equipment", sort=False):
        category = g["Category"].iloc[0]
        pphg_group = g["PPHG Group"].iloc[0] if "PPHG Group" in g.columns else category

        # Testo measurement points can have different sampling cadences.
        # Continuity must therefore use this equipment's own cadence rather
        # than the global median, otherwise a valid WCH/UDC/etc. sequence can
        # be incorrectly split into hundreds of single-point events.
        local_deltas = (
            g["Timestamp"].sort_values().diff().dropna()
        )
        local_deltas = local_deltas[local_deltas > pd.Timedelta(0)]
        local_interval = (
            local_deltas.median()
            if not local_deltas.empty
            else median_interval
        )

        stats = excursion_stats(g, pphg_group, local_interval)

        if pphg_group in RULES:
            rule = RULES[pphg_group]
            exceeded = max(stats["duration"] - rule["delay"], pd.Timedelta(0))
            limit_text = f"≥{rule['limit']:g}°C / {rule['delay'].total_seconds()/3600:g}h"
        else:
            exceeded = pd.Timedelta(0)
            limit_text = "N/A"

        rows.append({
            "Equipment": equipment,
            "Category": category,
            "PPHG Group": pphg_group,
            "Min °C": g["Temperature"].min(),
            "Average °C": g["Temperature"].mean(),
            "Max °C": g["Temperature"].max(),
            "Alarm Limit": limit_text,
            "Longest Start": stats["start"],
            "Longest End": stats["end"],
            "Longest Continuous": stats["duration"],
            "Exceeded By": exceeded,
            "Status": stats["status"],
            "Peak During Excursion °C": stats["peak"],
            "Threshold Events": stats["threshold_events"],
        })

    out = pd.DataFrame(rows)
    order = {"ALARM": 0, "WARNING": 1, "NORMAL": 2, "N/A": 3}
    out["_order"] = out["Status"].map(order).fillna(9)
    out = out.sort_values(
        ["_order", "Longest Continuous"],
        ascending=[True, False]
    ).drop(columns="_order")
    return out, median_interval





def _report_footer(canvas, doc):
    canvas.saveState()
    w, h = landscape(A4)
    canvas.setStrokeColor(colors.HexColor("#D7DEE5"))
    canvas.line(12*mm, 8*mm, w-12*mm, 8*mm)
    canvas.setFont("Helvetica", 6.5)
    canvas.setFillColor(colors.HexColor("#6B7785"))
    canvas.drawString(12*mm, 4.5*mm, "PPHG Temperature Analyzer • Analytical screening report")
    canvas.drawRightString(w-12*mm, 4.5*mm, f"Page {doc.page}")
    canvas.restoreState()


def pdf_status_chart(result):
    """Status distribution used in the PDF. Zero-count N/A is not displayed."""
    order = [
        ("Alarm", "ALARM", "#E84A5F"),
        ("Warning", "WARNING", "#FF8A4C"),
        ("Normal", "NORMAL", "#4CCB88"),
        ("Single Point", "SINGLE POINT", "#F4D35E"),
    ]
    active = [
        (label, int((result["Status"] == key).sum()), colors.HexColor(color))
        for label, key, color in order
        if int((result["Status"] == key).sum()) > 0
    ]
    d = Drawing(300, 135)
    pie = Pie()
    pie.x, pie.y, pie.width, pie.height = 12, 8, 112, 112
    pie.data = [x[1] for x in active]
    pie.labels = ["" for _ in active]
    for i, (_, _, fill) in enumerate(active):
        pie.slices[i].fillColor = fill
        pie.slices[i].strokeColor = colors.white
        pie.slices[i].strokeWidth = 0.7
    d.add(pie)

    from reportlab.graphics.shapes import String, Rect
    y = 103
    for label, count, fill in active:
        d.add(Rect(150, y-3, 9, 9, fillColor=fill,
                   strokeColor=colors.HexColor("#6B7785"), strokeWidth=0.35))
        d.add(String(166, y, f"{label}  {count}",
                     fontName="Helvetica", fontSize=8,
                     fillColor=colors.HexColor("#334155")))
        y -= 22
    return d

def pdf_status_count_chart(alarm_df):
    """Status count chart used in the PDF. N/A is intentionally omitted."""
    result = _CURRENT_RESULT_FOR_PDF
    order = [
        ("Alarm", "ALARM", "#E84A5F"),
        ("Warning", "WARNING", "#FF8A4C"),
        ("Normal", "NORMAL", "#4CCB88"),
        ("Single Point", "SINGLE POINT", "#F4D35E"),
    ]
    active = [
        (label, int((result["Status"] == key).sum()), colors.HexColor(color))
        for label, key, color in order
        if int((result["Status"] == key).sum()) > 0
    ]

    from reportlab.graphics.shapes import String, Rect, Line
    d = Drawing(405, 135)
    left, right = 44, 390
    bottom, top = 20, 110
    max_count = max([x[1] for x in active] or [1])
    grid_max = max(10, ((max_count + 1) // 2) * 2)

    for tick in range(0, grid_max + 1, 2):
        y = bottom + (top-bottom) * tick / grid_max
        d.add(Line(left, y, right, y,
                   strokeColor=colors.HexColor("#DDE3E8"), strokeWidth=0.45))
        d.add(String(left-6, y-2.5, str(tick),
                     fontName="Helvetica", fontSize=6.5,
                     textAnchor="end", fillColor=colors.HexColor("#667481")))

    slot = (right-left) / max(1, len(active))
    bar_w = min(52, slot*0.62)
    for i, (label, count, fill) in enumerate(active):
        cx = left + slot*(i+0.5)
        h = (top-bottom)*count/grid_max
        d.add(Rect(cx-bar_w/2, bottom, bar_w, h, fillColor=fill, strokeColor=None))
        d.add(String(cx, bottom+h+5, str(count),
                     fontName="Helvetica", fontSize=7.2,
                     textAnchor="middle", fillColor=colors.HexColor("#667481")))
        d.add(String(cx, 7, label,
                     fontName="Helvetica", fontSize=6.8,
                     textAnchor="middle", fillColor=colors.HexColor("#667481")))
    return d

def build_pdf_report(result, data, median_interval, raw=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        rightMargin=10*mm, leftMargin=10*mm,
        topMargin=10*mm, bottomMargin=11*mm,
        title="PPHG Temperature Analysis Report",
        author="PPHG Temperature Analyzer",
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=19, leading=22,
        textColor=colors.HexColor("#17324D"), alignment=TA_LEFT, spaceAfter=3*mm
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"], fontSize=13.5, leading=16,
        textColor=colors.HexColor("#17324D"), spaceBefore=1*mm, spaceAfter=3*mm
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=10.5, leading=13,
        textColor=colors.HexColor("#17324D"), spaceBefore=3*mm, spaceAfter=2*mm
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=8.2, leading=10.5, spaceAfter=1.5*mm
    )
    small = ParagraphStyle(
        "Small", parent=styles["BodyText"], fontSize=7.1, leading=8.6
    )
    tiny = ParagraphStyle(
        "Tiny", parent=styles["BodyText"], fontSize=6.5, leading=7.8
    )
    note = ParagraphStyle(
        "Note", parent=styles["BodyText"], fontSize=7.7, leading=9.5,
        textColor=colors.HexColor("#566574")
    )
    callout = ParagraphStyle(
        "Callout", parent=body, fontSize=8.5, leading=11,
        backColor=colors.HexColor("#EEF5FA"),
        borderColor=colors.HexColor("#C7DCEB"), borderWidth=0.5, borderPadding=6
    )

    alarms = int((result["Status"] == "ALARM").sum())
    warnings = int((result["Status"] == "WARNING").sum())
    normal = int((result["Status"] == "NORMAL").sum())
    other = int((result["Status"] == "N/A").sum())
    single = int((result["Status"] == "SINGLE POINT").sum())
    urgent = int((
        result.loc[result["Status"] == "ALARM", "Longest Continuous"]
        > pd.Timedelta(hours=24)
    ).sum())

    analysis_start = data["Timestamp"].min()
    analysis_end = data["Timestamp"].max()
    period = (
        f"{analysis_start.strftime('%d %b %Y %H:%M')} – {analysis_end.strftime('%d %b %Y %H:%M')}"
        if pd.notna(analysis_start) and pd.notna(analysis_end) else "—"
    )

    alarm_df = result[result["Status"] == "ALARM"].sort_values(
        ["Exceeded By", "Longest Continuous"], ascending=[False, False]
    ).copy()
    warning_df = result[result["Status"] == "WARNING"].sort_values(
        ["Longest Continuous", "Threshold Events"], ascending=[False, False]
    ).copy()
    single_df = result[result["Status"] == "SINGLE POINT"].copy()
    repeat = result[result["Threshold Events"] >= 2].sort_values(
        ["Threshold Events", "Longest Continuous"], ascending=[False, False]
    ).copy()
    # Any ALARM with a continuous event >24h is an urgent priority unit.
    # Keep recurrence definition intact, while separately identifying the
    # subset whose urgent condition comes from a single threshold event.
    urgent_over24 = result[
        (result["Status"] == "ALARM")
        & (result["Longest Continuous"] > pd.Timedelta(hours=24))
    ].sort_values("Longest Continuous", ascending=False).copy()
    urgent_single = urgent_over24[urgent_over24["Threshold Events"] < 2].copy()

    story = []

    # PAGE 1 — Executive Summary
    story.append(Paragraph("1. Executive Summary", title))
    story.append(Paragraph(
        f"<b>Overall Assessment: {'ATTENTION REQUIRED' if alarms else ('MONITOR' if warnings else 'NORMAL')}</b> "
        f"• Analysis period: <b>{period}</b> • Sampling interval: approximately "
        f"<b>{round(median_interval.total_seconds()/60):g} min</b>",
        callout
    ))

    # Keep KPI cards clearly below the assessment banner.
    story.append(Spacer(1, 4*mm))
    # KPI cards: use a dedicated leading so the 8pt label and 17pt
    # value have enough vertical line spacing. The previous shared body
    # leading was too small and caused the value to overlap the label.
    kpi_text = ParagraphStyle(
        "KPICardText",
        parent=body,
        fontSize=8.2,
        leading=16,
        spaceAfter=0,
        spaceBefore=0,
    )
    kpi_cells = [
        Paragraph(f"<b>Equipment</b><br/><font size=17 color='#17324D'>{len(result)}</font>", kpi_text),
        Paragraph(f"<b>Alarm</b><br/><font size=17 color='#E84A5F'>{alarms}</font>", kpi_text),
        Paragraph(f"<b>Warning</b><br/><font size=17 color='#FF8A4C'>{warnings}</font>", kpi_text),
        Paragraph(f"<b>Normal</b><br/><font size=17 color='#4CCB88'>{normal}</font>", kpi_text),
        Paragraph(f"<b>Single Point</b><br/><font size=17 color='#B38B00'>{single}</font>", kpi_text),
    ]
    kpi = Table([kpi_cells], colWidths=[53.4*mm]*5, rowHeights=[16.5*mm])
    kpi.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,0),colors.HexColor("#F3F7FA")),
        ("BACKGROUND",(1,0),(1,0),colors.HexColor("#FFF0F0")),
        ("BACKGROUND",(2,0),(2,0),colors.HexColor("#FFF4E6")),
        ("BACKGROUND",(3,0),(3,0),colors.HexColor("#EAF8F0")),
        ("BACKGROUND",(4,0),(4,0),colors.HexColor("#FFF8D9")),
        ("BOX",(0,0),(-1,-1),0.35,colors.HexColor("#D0D7DE")),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#D0D7DE")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(0,0),(-1,-1),"LEFT"),
        ("LEFTPADDING",(0,0),(-1,-1),5),
        ("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))
    story.append(kpi)
    story.append(Paragraph(
        "<b>Single Point</b> = threshold observed but no elapsed duration can be established.",
        note
    ))
    story.append(Spacer(1,4*mm))

    global _CURRENT_RESULT_FOR_PDF
    _CURRENT_RESULT_FOR_PDF = result

    # Give the two dashboard charts equal, generous columns. Keep headings
    # separate from the chart box to avoid any visual collision.
    chart_labels = Table([[
        Paragraph("<b>Status Distribution</b>", h2),
        Paragraph("<b>Status Count</b>", h2),
    ]], colWidths=[125*mm,125*mm], rowHeights=[7*mm])
    chart_labels.setStyle(TableStyle([
        ("LEFTPADDING",(0,0),(-1,-1),3), ("RIGHTPADDING",(0,0),(-1,-1),3),
        ("TOPPADDING",(0,0),(-1,-1),0), ("BOTTOMPADDING",(0,0),(-1,-1),0),
        ("VALIGN",(0,0),(-1,-1),"BOTTOM"),
    ]))
    story.append(chart_labels)

    charts = Table([[pdf_status_chart(result), pdf_status_count_chart(alarm_df)]],
                   colWidths=[133.5*mm,133.5*mm], rowHeights=[43*mm])
    charts.setStyle(TableStyle([
        ("BOX",(0,0),(-1,-1),0.45,colors.HexColor("#D0D7DE")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2),
        ("TOPPADDING",(0,0),(-1,-1),1),("BOTTOMPADDING",(0,0),(-1,-1),1),
    ]))
    story.append(charts)
    story.append(Spacer(1,3*mm))

    story.append(Paragraph("Priority Overview — Continuous Excursions >24h", h2))
    priority_rows = [["Rank","Equipment","Status","Continuous","Exceeded By","Peak °C"]]
    for rank, (_, r) in enumerate(urgent_over24.iterrows(), 1):
        priority_rows.append([
            str(rank), Paragraph(str(r["Equipment"]), small),
            "URGENT >24h" if r["Longest Continuous"] > pd.Timedelta(hours=24) else "REVIEW",
            format_duration(r["Longest Continuous"]),
            format_duration(r["Exceeded By"]),
            f"{r['Peak During Excursion °C']:.1f}",
        ])
    pt = Table(priority_rows, colWidths=[13*mm,100*mm,34*mm,31*mm,31*mm,20*mm], repeatRows=1)
    pt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324D")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),7.5),
        ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#C5CCD3")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(0,1),(0,-1),"CENTER"),("ALIGN",(2,1),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),1.7),("BOTTOMPADDING",(0,0),(-1,-1),1.7),
    ]))
    for i, (_, r) in enumerate(urgent_over24.iterrows(), 1):
        if r["Longest Continuous"] > pd.Timedelta(hours=24):
            pt.setStyle(TableStyle([("BACKGROUND",(0,i),(-1,i),colors.HexColor("#FFF0F0"))]))
    story.append(pt)

    story.append(Paragraph("PPHG Criteria Used", h2))
    criteria = [
        ["Chiller", "≥ 6°C", "2 hours", "Continuous threshold event ≥2h → ALARM"],
        ["Freezer", "≥ -15°C", "4 hours", "Continuous threshold event ≥4h → ALARM"],
    ]
    ct = Table(criteria, colWidths=[32*mm,32*mm,32*mm,124*mm])
    ct.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#F6F8FA")),
        ("BOX",(0,0),(-1,-1),0.35,colors.HexColor("#C5CCD3")),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#D9E0E6")),
        ("FONTNAME",(0,0),(-1,-1),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1),7.3),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))
    story.append(ct)
    # PAGE 2 — Equipment Analysis
    story.append(PageBreak())
    story.append(Paragraph("2. Equipment Analysis", title))
    story.append(Paragraph(
        f"<b>{len(result)} equipment</b> • Sorted by status and longest continuous threshold event. "
        "Status colors are applied only to the final column for quick scanning.",
        note
    ))
    story.append(Paragraph(
        "<b>Reading guide:</b> Continuous = longest threshold event. "
        "Exceeded = time beyond the applicable PPHG delay. "
        "Threshold Events = number of distinct periods detected at or above the applicable PPHG threshold. This is a recurrence indicator, not a failure count.",
        note
    ))
    story.append(Spacer(1,1.5*mm))

    # Keep long equipment/category names readable. Paragraphs allow safe
    # wrapping instead of text colliding with adjacent columns.
    header_style = ParagraphStyle(
        "EquipmentTableHeader",
        parent=tiny,
        fontName="Helvetica-Bold",
        fontSize=6.6,
        leading=7.2,
        textColor=colors.white,
        alignment=0,
    )
    cell_style = ParagraphStyle(
        "EquipmentTableCell",
        parent=tiny,
        fontSize=6.8,
        leading=8.0,
        spaceAfter=0,
        spaceBefore=0,
    )

    headers = [
        Paragraph("Equipment", header_style),
        Paragraph("Category", header_style),
        Paragraph("Min °C", header_style),
        Paragraph("Avg °C", header_style),
        Paragraph("Max °C", header_style),
        Paragraph("Start", header_style),
        Paragraph("End", header_style),
        Paragraph("Continuous", header_style),
        Paragraph("Exceeded", header_style),
        Paragraph("Threshold<br/>Events", header_style),
        Paragraph("Status", header_style),
    ]
    rows = [headers]
    for _, r in result.iterrows():
        start = r["Longest Start"].strftime("%d-%m %H:%M") if pd.notna(r["Longest Start"]) else "—"
        end = r["Longest End"].strftime("%d-%m %H:%M") if pd.notna(r["Longest End"]) else "—"
        rows.append([
            Paragraph(str(r["Equipment"]), cell_style),
            Paragraph(str(r["Category"]), cell_style),
            f"{r['Min °C']:.1f}", f"{r['Average °C']:.1f}", f"{r['Max °C']:.1f}",
            start, end, format_duration(r["Longest Continuous"]),
            format_duration(r["Exceeded By"]) if r["Exceeded By"] > pd.Timedelta(0) else "—",
            str(int(r["Threshold Events"])),
            str(r["Status"])
        ])

    # Same overall table footprint, but more room for Equipment/Category and
    # enough width for time columns. Long labels wrap within their own cell.
    at = Table(
        rows,
        colWidths=[
            62*mm, 27*mm, 12*mm, 12*mm, 12*mm,
            21*mm, 21*mm, 25*mm, 23*mm, 19*mm, 18*mm
        ],
        repeatRows=1,
    )
    cmd = [
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324D")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,1),(-1,-1),6.8),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#C5CCD3")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ALIGN",(2,1),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),2.0),("BOTTOMPADDING",(0,0),(-1,-1),2.0),
    ]
    for i, (_, r) in enumerate(result.iterrows(), 1):
        status = r["Status"]
        if status == "ALARM":
            cmd += [("BACKGROUND",(-1,i),(-1,i),colors.HexColor("#FFD8D8")),
                    ("TEXTCOLOR",(-1,i),(-1,i),colors.HexColor("#B00020")),
                    ("FONTNAME",(-1,i),(-1,i),"Helvetica-Bold")]
        elif status == "WARNING":
            cmd.append(("BACKGROUND",(-1,i),(-1,i),colors.HexColor("#FFF0CC")))
        elif status == "NORMAL":
            cmd.append(("BACKGROUND",(-1,i),(-1,i),colors.HexColor("#D9F5E5")))
        elif status == "SINGLE POINT":
            cmd.append(("BACKGROUND",(-1,i),(-1,i),colors.HexColor("#FFF7D6")))
    at.setStyle(TableStyle(cmd))
    story.append(at)

    # PAGE 3 — Alarm analysis and priority
    story.append(PageBreak())
    story.append(Paragraph("3. Alarm Analysis & Priority Review", title))
    story.append(Paragraph(
        f"<b>{alarms} ALARM equipment</b> identified. "
        f"<b>{urgent}</b> have continuous threshold events &gt;24h and are classified <b>REVIEW URGENTLY</b>.",
        callout
    ))
    story.append(Spacer(1, 3*mm))
    priority = [["Rank","Equipment","Category","Continuous","Exceeded By","Peak °C","Threshold Events"]]
    for rank, (_, r) in enumerate(alarm_df.iterrows(), 1):
        priority.append([
            str(rank), Paragraph(str(r["Equipment"]), small), str(r["Category"]),
            format_duration(r["Longest Continuous"]),
            format_duration(r["Exceeded By"]),
            f"{r['Peak During Excursion °C']:.1f}°C",
            str(int(r["Threshold Events"]))
        ])
    pt2 = Table(priority, colWidths=[13*mm,91*mm,25*mm,34*mm,34*mm,25*mm,28*mm], repeatRows=1)
    st2 = [
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324D")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),7.3),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#C5CCD3")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ALIGN",(0,1),(0,-1),"CENTER"),
        ("ALIGN",(2,1),(-1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),3.2),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]
    for i, (_, r) in enumerate(alarm_df.iterrows(),1):
        if r["Longest Continuous"] > pd.Timedelta(hours=24):
            st2.append(("BACKGROUND",(0,i),(-1,i),colors.HexColor("#FFF0F0")))
    pt2.setStyle(TableStyle(st2))
    story.append(pt2)
    story.append(Spacer(1,4*mm))
    story.append(Paragraph(
        "<b>Priority interpretation:</b> An ALARM unit with a continuous threshold event exceeding 24 hours may indicate a sustained equipment performance issue "
        "and should receive priority operational review. This is an analytical flag, not a root-cause diagnosis.",
        note
    ))
    story.append(Paragraph(
        "<b>General review notes:</b> For chillers, review door/loading practices, ambient exposure, condenser/coil condition, airflow and temperature control. "
        "For freezers, review refrigeration performance, door/seal condition, defrost operation, loading and condenser/coil condition. "
        "Repeated threshold events also indicate a recurrence pattern that may warrant operational review.",
        note
    ))

    # PAGE 4 — Warning + single point
    story.append(PageBreak())
    story.append(Paragraph("4. Warning & Single-Point Review", title))
    if warnings:
        story.append(Paragraph("WARNING — Monitor & Follow Up", h2))
        wd = [["Equipment","Category","Continuous","Threshold Events","Peak °C"]]
        for _, r in warning_df.iterrows():
            wd.append([
                Paragraph(str(r["Equipment"]), small), str(r["Category"]),
                format_duration(r["Longest Continuous"]), str(int(r["Threshold Events"])),
                f"{r['Peak During Excursion °C']:.1f}"
            ])
        wt = Table(wd, colWidths=[92*mm,30*mm,36*mm,30*mm,28*mm], repeatRows=1)
        wt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#FFF0CC")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#6B4A00")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#C5CCD3")),
            ("FONTSIZE",(0,0),(-1,-1),7.0),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("ALIGN",(1,1),(-1,-1),"CENTER"),
            ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ]))
        story.append(wt)
        story.append(Paragraph(
            "<b>Warning interpretation:</b> These units exceeded the applicable temperature threshold but did not reach the PPHG alarm duration. "
            "They should be monitored for recurrence, especially when event duration approaches the alarm delay.",
            note
        ))
        story.append(Paragraph(
            "<b>General review notes:</b> Chillers — door/loading, ambient exposure, condenser/coil, airflow and temperature control. "
            "Freezers — refrigeration performance, door/seal, defrost, loading and condenser/coil.",
            note
        ))

    if not single_df.empty:
        story.append(Spacer(1,3*mm))
        story.append(Paragraph("Single-Point Threshold Events", h2))
        story.append(Paragraph(
            "A threshold event was observed, but no elapsed duration can be established from the available point(s). "
            "These are not classified as WARNING or ALARM.",
            note
        ))
        sp = [["Equipment","Category","Observed °C","Timestamp"]]
        for _, r in single_df.iterrows():
            ts = r["Longest Start"].strftime("%d-%m-%Y %H:%M") if pd.notna(r["Longest Start"]) else "—"
            sp.append([Paragraph(str(r["Equipment"]), tiny), str(r["Category"]),
                       f"{r['Peak During Excursion °C']:.1f}", ts])
        stp = Table(sp, colWidths=[85*mm,28*mm,30*mm,60*mm], repeatRows=1)
        stp.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#FFF7D6")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#7A5A00")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#C5CCD3")),
            ("FONTSIZE",(0,0),(-1,-1),7),("ALIGN",(1,1),(-1,-1),"CENTER"),
        ]))
        story.append(stp)

    # PAGE 5 — Threshold Recurrence
    story.append(PageBreak())
    story.append(Paragraph("5. Threshold Recurrence", title))
    story.append(Paragraph(
        f"<b>{len(repeat)}</b> equipment recorded two or more distinct threshold events during the analysis period. "
        f"Separately, <b>{len(urgent_over24)}</b> equipment had a continuous threshold event exceeding 24 hours; "
        f"<b>{len(urgent_single)}</b> of these had only one threshold event. "
        "Recurrence frequency and prolonged excursions are shown separately.",
        note
    ))

    if not repeat.empty:
        story.append(Paragraph("Repeated Threshold Events", h2))
        rd = [["Equipment", "Category", "Threshold Events", "Longest Event", "Status"]]
        for _, r in repeat.iterrows():
            rd.append([
                Paragraph(str(r["Equipment"]), tiny),
                str(r["Category"]),
                str(int(r["Threshold Events"])),
                format_duration(r["Longest Continuous"]),
                str(r["Status"]),
            ])
        rt = Table(
            rd,
            colWidths=[105*mm,25*mm,28*mm,35*mm,36*mm],
            repeatRows=1
        )
        rt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EAF2F8")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#17324D")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#C5CCD3")),
            ("FONTSIZE",(0,0),(-1,-1),6.6),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("ALIGN",(1,1),(-1,-1),"CENTER"),
            ("TOPPADDING",(0,0),(-1,-1),1.6),
            ("BOTTOMPADDING",(0,0),(-1,-1),1.6),
        ]))
        story.append(rt)

    if not urgent_single.empty:
        story.append(Spacer(1,2.5*mm))
        story.append(Paragraph("Urgent >24h — Single-Event Exceptions", h2))
        ud = [["Equipment", "Category", "Longest Event", "Exceeded By", "Peak °C"]]
        for _, r in urgent_single.iterrows():
            ud.append([
                Paragraph(str(r["Equipment"]), tiny),
                str(r["Category"]),
                format_duration(r["Longest Continuous"]),
                format_duration(r["Exceeded By"]),
                f"{r['Peak During Excursion °C']:.1f}",
            ])
        ut = Table(
            ud,
            colWidths=[105*mm,25*mm,35*mm,35*mm,29*mm],
            repeatRows=1
        )
        ut.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#FFF0F0")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#B00020")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#C5CCD3")),
            ("FONTSIZE",(0,0),(-1,-1),6.8),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("ALIGN",(1,1),(-1,-1),"CENTER"),
            ("TOPPADDING",(0,0),(-1,-1),2),
            ("BOTTOMPADDING",(0,0),(-1,-1),2),
        ]))
        story.append(ut)

    story.append(Spacer(1,2.5*mm))
    story.append(Paragraph(
        "<b>How to read this:</b> A threshold event is one distinct period where the measured temperature is at or above the applicable PPHG threshold. "
        "A high event count indicates frequent threshold crossing; it does <b>not</b> mean the equipment failed that many times. "
        "Event count should be interpreted together with Longest Continuous, Peak temperature and operating records.",
        note
    ))

    story.append(Spacer(1,4*mm))
    story.append(Paragraph(
        "<b>Report note:</b> This report is an analytical screening based on uploaded Testo measurement data and the configured PPHG thresholds. "
        "Review findings together with operational records, maintenance history and applicable SOP requirements before formal corrective action or conclusions.",
        note
    ))

    doc.build(story, onFirstPage=_report_footer, onLaterPages=_report_footer)
    buffer.seek(0)
    return buffer.getvalue()



uploaded = st.file_uploader("Upload Testo CSV", type=["csv"], accept_multiple_files=True)

if uploaded:
    try:
        data, raw, source_files = parse_testo(uploaded)
        result, median_interval = analyze(data)

        equipment_count = result["Equipment"].nunique()
        st.success(
            f"Testo CSV berhasil dibaca: {equipment_count} equipment dianalisis. "
            f"Sampling interval terdeteksi ≈ {round(median_interval.total_seconds()/60):g} menit."
        )

        unmapped = sorted(
            result.loc[result["Status"] == "N/A", "Equipment"].dropna().unique().tolist()
        )
        if unmapped:
            st.info(
                f"{len(unmapped)} equipment memiliki kode kategori yang belum dikenal: "
                "tetap dianalisis sebagai Other/N/A: " + ", ".join(unmapped)
            )

        alarms = int((result["Status"] == "ALARM").sum())
        warnings = int((result["Status"] == "WARNING").sum())
        normal = int((result["Status"] == "NORMAL").sum())
        single_status = int((result["Status"] == "SINGLE POINT").sum())
        na_status = int((result["Status"] == "N/A").sum())

        # ------------------------------------------------------------
        # Executive Summary
        # ------------------------------------------------------------
        st.subheader("Executive Summary")
        st.caption(
            "Overview of equipment status based on PPHG temperature threshold-event analysis."
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Equipment", equipment_count)
        c2.metric("🔴 Alarm", alarms)
        c3.metric("🟠 Warning", warnings)
        c4.metric("🟢 Normal", normal)
        c5.metric("🟡 Single Point", single_status)

        chart_left, chart_right = st.columns(2)

        # Status distribution donut.
        status_df = pd.DataFrame({
            "Status": ["Alarm", "Warning", "Normal", "Single Point"],
            "Count": [alarms, warnings, normal, single_status],
        })
        status_df = status_df[status_df["Count"] > 0]

        with chart_left:
            st.markdown("**Status Distribution**")
            donut = px.pie(
                status_df,
                names="Status",
                values="Count",
                hole=0.58,
            )
            donut.update_traces(
                textposition="inside",
                textinfo="percent",
                hovertemplate="%{label}: %{value} equipment<extra></extra>",
                marker=dict(
                    colors=["#E84A5F", "#FF8A4C", "#4CCB88", "#9AA3AD"]
                ),
            )
            donut.update_layout(
                margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="v"),
                showlegend=True,
            )
            st.plotly_chart(
                donut,
                width="stretch",
                config={"displayModeBar": False},
            )

        # Status count bar chart. This is a distribution, not a historical trend.
        with chart_right:
            st.markdown("**Status Count**")
            bar = px.bar(
                status_df,
                x="Status",
                y="Count",
                text="Count",
            )
            bar.update_traces(
                textposition="outside",
                hovertemplate="%{x}: %{y} equipment<extra></extra>",
                marker_color=[
                    {"Alarm": "#E84A5F", "Warning": "#FF8A4C",
                     "Normal": "#4CCB88", "Single Point": "#F4D35E", "N/A": "#9AA3AD"}[s]
                    for s in status_df["Status"]
                ],
            )
            bar.update_layout(
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title=None,
                yaxis_title="Equipment",
                showlegend=False,
            )
            st.plotly_chart(
                bar,
                width="stretch",
                config={"displayModeBar": False},
            )

        # Threshold-event recurrence: useful as a pattern indicator, not a failure count.
        repeated_df = result[result["Threshold Events"] >= 2].copy()
        urgent_single_df = result[
            (result["Status"] == "ALARM")
            & (result["Longest Continuous"] > pd.Timedelta(hours=24))
            & (result["Threshold Events"] < 2)
        ].copy()
        st.markdown("**Threshold Recurrence**")
        a1, a2, a3 = st.columns(3)
        urgent_over24_df = result[
            (result["Status"] == "ALARM")
            & (result["Longest Continuous"] > pd.Timedelta(hours=24))
        ].copy()
        a1.metric("Units with Repeated Events", len(repeated_df))
        a2.metric("Urgent >24h", len(urgent_over24_df))
        a3.metric("Total Threshold Events", int(result["Threshold Events"].sum()))
        if not urgent_single_df.empty:
            st.info(
                f"**{len(urgent_single_df)} of {len(urgent_over24_df)} urgent >24h units** have only one threshold event. "
                "A single prolonged event is a priority condition, not a recurrence pattern."
            )

        if not repeated_df.empty:
            top_repeat = repeated_df.sort_values(
                ["Threshold Events", "Longest Continuous"],
                ascending=[False, False]
            ).iloc[0]
            st.info(
                f"**Highest event count:** {top_repeat['Equipment']} recorded "
                f"**{int(top_repeat['Threshold Events'])} distinct threshold events**. "
                "This indicates frequent threshold crossing and should not be interpreted as a count of equipment failures."
            )

        # Management-oriented findings: concise, data-derived and actionable.
        st.markdown("**Key Findings**")
        findings = []

        if alarms:
            alarm_df = result[result["Status"] == "ALARM"].copy()
            alarm_df = alarm_df.sort_values(
                ["Exceeded By", "Longest Continuous"],
                ascending=[False, False],
            )
            urgent = int(
                (
                    alarm_df["Longest Continuous"]
                    > pd.Timedelta(hours=24)
                ).sum()
            )
            findings.append(
                f"**{alarms} equipment** meet the PPHG alarm-duration criterion."
            )
            if urgent:
                findings.append(
                    f"**{urgent} equipment** have a continuous threshold event exceeding "
                    f"24 hours and should receive priority operational review."
                )

            top = alarm_df.iloc[0]
            findings.append(
                f"Longest threshold event: **{top['Equipment']}** — "
                f"{format_duration(top['Longest Continuous'])}, "
                f"peak **{top['Peak During Excursion °C']:.1f}°C**."
            )

        if warnings:
            findings.append(
                f"**{warnings} equipment** recorded threshold events below the PPHG "
                f"alarm-duration threshold and should be monitored for recurrence."
            )

        if single_status:
            findings.append(
                f"**{single_status} equipment** have threshold observations without an established elapsed duration and are classified as **SINGLE POINT**."
            )
        if not findings:
            findings.append("No PPHG Chiller/Freezer alarm-duration threshold events were identified.")

        for finding in findings:
            st.markdown(f"• {finding}")

        st.divider()
        st.subheader("PPHG Analysis")

        # Keep the analysis filters here. They do not affect the Executive Summary;
        # they only control which rows are shown in the detailed analysis table.
        f1, f2 = st.columns(2)
        # Keep the UI simple: filter by the two PPHG groups only.
        # Detailed equipment type remains visible in the Category column.
        category_options = ["All", "Chiller", "Freezer"]
        status_options = ["All", "ALARM", "WARNING", "NORMAL", "SINGLE POINT"]

        selected_category = f1.selectbox(
            "Filter by category",
            category_options,
            key="analysis_category_filter",
        )
        selected_status = f2.selectbox(
            "Filter by status",
            status_options,
            key="analysis_status_filter",
        )
        st.caption("Category filter: Chiller / Freezer. Detailed equipment type remains shown in the Category column.")

        filtered_result = result.copy()
        if selected_category != "All":
            filtered_result = filtered_result[
                filtered_result["PPHG Group"] == selected_category
            ]
        if selected_status != "All":
            filtered_result = filtered_result[
                filtered_result["Status"] == selected_status
            ]

        st.caption(
            f"Showing **{len(filtered_result)} of {len(result)} equipment**."
        )

        # PPHG criteria are shown as highlights instead of a table column.
        st.markdown(
            """
            **PPHG Alarm Criteria:** 🔴 **Chiller ≥ 6°C for 2 hours** &nbsp;|&nbsp;
            🧊 **Freezer ≥ -15°C for 4 hours**

            **Longest Continuous** = longest continuous threshold event at or above the applicable
            temperature limit. **Exceeded By** = time beyond the applicable PPHG alarm delay.
            """,
            unsafe_allow_html=False,
        )

        show = filtered_result.copy()
        for col in ["Min °C", "Average °C", "Max °C", "Peak During Excursion °C"]:
            show[col] = show[col].map(
                lambda x: f"{x:.1f}" if pd.notna(x) else "—"
            )
        for col in ["Longest Start", "Longest End"]:
            show[col] = pd.to_datetime(
                show[col], errors="coerce"
            ).dt.strftime("%d-%m-%Y %H:%M").fillna("—")
        show["Longest Continuous"] = show["Longest Continuous"].map(format_duration)
        show["Exceeded By"] = show["Exceeded By"].map(
            lambda x: format_duration(x) if x > pd.Timedelta(0) else "—"
        )

        table_cols = [
            "Equipment", "Category", "Min °C", "Average °C", "Max °C",
            "Longest Start", "Longest End",
            "Longest Continuous", "Exceeded By", "Threshold Events", "Status"
        ]

        def status_style(row):
            styles = [""] * len(row)
            i = table_cols.index("Status")
            if row["Status"] == "ALARM":
                styles[i] = "background-color:#ffd6d6;color:#b00020;font-weight:700"
            elif row["Status"] == "WARNING":
                styles[i] = "background-color:#fff0cc;color:#9a5b00;font-weight:700"
            elif row["Status"] == "NORMAL":
                styles[i] = "background-color:#d9f5e5;color:#087443;font-weight:700"
            return styles

        st.dataframe(
            show[table_cols].style.apply(status_style, axis=1),
            width="stretch",
            hide_index=True,
        )

        st.divider()
        st.subheader("Equipment Detail")

        if filtered_result.empty:
            st.info("No equipment matches the selected filters.")
            st.stop()

        selected = st.selectbox(
            "Select equipment",
            filtered_result["Equipment"].tolist()
        )

        selected_result = result[result["Equipment"] == selected].iloc[0]
        trend = data[data["Equipment"] == selected].copy()

        fig = px.line(
            trend,
            x="Timestamp",
            y="Temperature",
            title=f"{selected} — Temperature Trend",
        )

        category = selected_result["Category"]
        if category in RULES:
            limit = RULES[category]["limit"]
            fig.add_hline(
                y=limit,
                line_dash="dash",
                annotation_text=f"PPHG limit {limit:g}°C"
            )

        st.plotly_chart(fig, width="stretch")

        d1, d2, d3 = st.columns(3)
        d1.metric("Longest Continuous", format_duration(selected_result["Longest Continuous"]))
        d2.metric(
            "Longest Event Start",
            selected_result["Longest Start"].strftime("%d-%m-%Y %H:%M")
            if pd.notna(selected_result["Longest Start"]) else "—"
        )
        d3.metric(
            "Longest Event End",
            selected_result["Longest End"].strftime("%d-%m-%Y %H:%M")
            if pd.notna(selected_result["Longest End"]) else "—"
        )
        pdf_bytes = build_pdf_report(result, data, median_interval, raw=raw)
        st.download_button(
            "📄 Download PPHG PDF Report",
            pdf_bytes,
            "PPHG_Temperature_Analysis_Report.pdf",
            "application/pdf",
        )

    except Exception as e:
        st.error(f"CSV tidak dapat diproses: {e}")
