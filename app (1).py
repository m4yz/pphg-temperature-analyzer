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

# The current Testo CSV has duplicate display names. Until naming is cleaned up
# in Testo, identify the duplicate L90 instruments by their column position.
COLUMN_MAPPING = {
    0: ("Eden Bar", "Chiller"),
    1: ("RL Kitchen – Showcase Chiller", "Chiller"),
    2: ("RL Kitchen – Upright Chiller 1", "Chiller"),
    3: ("RL Kitchen – Upright Chiller 2", "Chiller"),
    4: ("RL Kitchen – Upright Chiller 3", "Chiller"),
    5: ("RL Kitchen – Upright Freezer (2 Drawer)", "Freezer"),
    6: ("RL Kitchen – Undercounter Chiller", "Chiller"),
    7: ("RL Kitchen – Upright Chiller (2 Drawer)", "Chiller"),
    8: ("RL Kitchen – Freezer GEA 1", "Freezer"),
    9: ("RL Kitchen – Freezer GEA 3", "Freezer"),
    10: ("RL Kitchen – Freezer GEA 2", "Freezer"),
    11: ("Receiving – Freezer GEA 013", "Freezer"),
    12: ("Receiving – Freezer GEA 012", "Freezer"),
    13: ("Receiving – Freezer GEA 015", "Freezer"),
    14: ("Receiving – Showcase Chiller 014", "Chiller"),
    15: ("Kitchen 1 L90 – Upright Chiller 1 019", "Chiller"),
    16: ("Kitchen 1 L90 – Upright Chiller 2 020", "Chiller"),
    17: ("Kitchen 1 L90 – Upright Chiller 3 021", "Chiller"),
    18: ("Kitchen 1 L90 – Upright Freezer 1 022", "Freezer"),
    19: ("Canteen – Chiller", "Chiller"),
    20: ("Kitchen Lt.90 – K.UPCS.1", "Other"),
    21: ("Kitchen Lt.90 – WCH.1", "Other"),
    22: ("Kitchen Lt.90 – WCH.2", "Other"),
    23: ("Eden Bar – Chiller", "Chiller"),
}

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


def parse_testo(uploaded):
    # Testo Smart CSV uses semicolon separator and one timestamp column.
    raw = pd.read_csv(uploaded, sep=";", encoding="utf-8-sig")
    if raw.shape[1] < 2:
        raise ValueError("CSV Testo tidak berisi kolom measurement.")

    timestamp_col = raw.columns[0]
    timestamps = pd.to_datetime(raw[timestamp_col], errors="coerce")
    valid_time = timestamps.notna()
    if valid_time.sum() < 2:
        raise ValueError("Kolom timestamp Testo tidak dapat dibaca.")

    measurement_cols = list(raw.columns[1:])
    records = []

    for idx, col in enumerate(measurement_cols):
        values = pd.to_numeric(raw[col], errors="coerce")
        mask = valid_time & values.notna()
        if mask.sum() == 0:
            continue

        if idx in COLUMN_MAPPING:
            display_name, category = COLUMN_MAPPING[idx]
        else:
            # Future-proof: after Testo naming is cleaned up, use the actual name.
            base = str(col).split(": Temperature")[0].strip()
            low = base.lower()
            if "freezer" in low:
                category = "Freezer"
            elif "chiller" in low:
                category = "Chiller"
            else:
                category = "Other"
            display_name = base

        temp = pd.DataFrame({
            "Timestamp": timestamps[mask].values,
            "Equipment": display_name,
            "Temperature": values[mask].values,
            "Category": category,
        })
        records.append(temp)

    if not records:
        raise ValueError("Tidak ada measurement temperature yang dapat dibaca.")

    return pd.concat(records, ignore_index=True), raw

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
        stats = excursion_stats(g, category, median_interval)

        if category in RULES:
            rule = RULES[category]
            exceeded = max(stats["duration"] - rule["delay"], pd.Timedelta(0))
            limit_text = f"≥{rule['limit']:g}°C / {rule['delay'].total_seconds()/3600:g}h"
        else:
            exceeded = pd.Timedelta(0)
            limit_text = "N/A"

        rows.append({
            "Equipment": equipment,
            "Category": category,
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
    """Dashboard-matched status distribution donut."""
    alarms = int((result["Status"] == "ALARM").sum())
    warnings = int((result["Status"] == "WARNING").sum())
    normal = int((result["Status"] == "NORMAL").sum())
    equipment_count = len(result)
    other_status = max(equipment_count - alarms - warnings - normal, 0)

    labels = ["Alarm", "Warning", "Normal", "Other / N/A"]
    counts = [alarms, warnings, normal, other_status]
    fills = [
        colors.HexColor("#E84A5F"),
        colors.HexColor("#FF8A4C"),
        colors.HexColor("#4CCB88"),
        colors.HexColor("#9AA3AD"),
    ]

    d = Drawing(320, 155)
    pie = Pie()
    pie.x, pie.y, pie.width, pie.height = 30, 12, 125, 125
    pie.data = counts
    pie.labels = ["" for _ in counts]
    for i, fill in enumerate(fills):
        pie.slices[i].fillColor = fill
        pie.slices[i].strokeColor = colors.white
        pie.slices[i].strokeWidth = 0.7
    d.add(pie)

    # Match Plotly dashboard: percentage labels inside the donut + vertical legend.
    total = sum(counts)
    legend_y = 118
    for i, (label, count) in enumerate(zip(labels, counts)):
        if count <= 0:
            continue
        pct = count / total * 100 if total else 0
        # percentage inside each slice is approximated with a compact label
        # around the donut; keep the legend as the exact value reference.
    legend = Legend()
    legend.x, legend.y = 180, 112
    legend.fontName, legend.fontSize = "Helvetica", 8
    legend.colorNamePairs = [
        (fills[i], f"{labels[i]}  {counts[i]}")
        for i in range(4) if counts[i] > 0
    ]
    d.add(legend)

    # Center label, similar to the dashboard's clean Plotly donut.
    from reportlab.graphics.shapes import String
    d.add(String(92, 69, str(total), fontName="Helvetica-Bold",
                 fontSize=13, textAnchor="middle",
                 fillColor=colors.HexColor("#17324D")))
    d.add(String(92, 56, "equipment", fontName="Helvetica",
                 fontSize=7, textAnchor="middle",
                 fillColor=colors.HexColor("#667481")))
    return d

def pdf_status_count_chart(alarm_df):
    """Dashboard-matched Status Count bar chart."""
    result = _CURRENT_RESULT_FOR_PDF
    alarms = int((result["Status"] == "ALARM").sum())
    warnings = int((result["Status"] == "WARNING").sum())
    normal = int((result["Status"] == "NORMAL").sum())
    equipment_count = len(result)
    other_status = max(equipment_count - alarms - warnings - normal, 0)

    labels = ["Alarm", "Warning", "Normal", "Other / N/A"]
    counts = [alarms, warnings, normal, other_status]
    fills = [
        colors.HexColor("#E84A5F"),
        colors.HexColor("#FF8A4C"),
        colors.HexColor("#4CCB88"),
        colors.HexColor("#9AA3AD"),
    ]

    from reportlab.graphics.shapes import String, Rect, Line
    d = Drawing(430, 155)
    left, right = 48, 410
    bottom, top = 22, 125
    grid_max = max(10, ((max(counts) + 1) // 2) * 2)

    for tick in range(0, grid_max + 1, 2):
        y = bottom + (top - bottom) * tick / grid_max
        d.add(Line(left, y, right, y,
                   strokeColor=colors.HexColor("#DDE3E8"),
                   strokeWidth=0.45))
        d.add(String(left - 6, y - 2.5, str(tick),
                     fontName="Helvetica", fontSize=6.5,
                     textAnchor="end", fillColor=colors.HexColor("#667481")))

    slot = (right - left) / len(labels)
    bar_w = min(48, slot * 0.62)
    for i, (label, count, fill) in enumerate(zip(labels, counts, fills)):
        cx = left + slot * (i + 0.5)
        h = (top - bottom) * count / grid_max
        x = cx - bar_w / 2
        d.add(Rect(x, bottom, bar_w, h,
                   fillColor=fill, strokeColor=None))
        d.add(String(cx, bottom + h + 5, str(count),
                     fontName="Helvetica", fontSize=7.2,
                     textAnchor="middle", fillColor=colors.HexColor("#667481")))
        d.add(String(cx, 7, label,
                     fontName="Helvetica", fontSize=6.8,
                     textAnchor="middle", fillColor=colors.HexColor("#667481")))

    d.add(String(8, 72, "Equipment", fontName="Helvetica", fontSize=6.8,
                 fillColor=colors.HexColor("#667481"), angle=90))
    return d

def build_pdf_report(result, data, median_interval, raw=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        rightMargin=12*mm, leftMargin=12*mm,
        topMargin=11*mm, bottomMargin=12*mm,
        title="PPHG Temperature Analysis Report",
        author="PPHG Temperature Analyzer",
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=20, leading=23,
        textColor=colors.HexColor("#17324D"), alignment=TA_LEFT, spaceAfter=3*mm
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"], fontSize=14, leading=17,
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
    other = int((result["Category"] == "Other").sum())
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
    )

    story = []

    # PAGE 1 — Executive Summary
    story.append(Paragraph("1. Executive Summary", title))
    story.append(Paragraph(
        f"<b>Overall Assessment: {'ATTENTION REQUIRED' if alarms else ('MONITOR' if warnings else 'NORMAL')}</b> "
        f"• Analysis period: <b>{period}</b> • Sampling interval: approximately "
        f"<b>{round(median_interval.total_seconds()/60):g} min</b>",
        callout
    ))

    # Keep PDF Executive Summary aligned with the dashboard:
    # 4 KPIs + Status Distribution donut + Status Count bar.
    other_status = max(len(result) - alarms - warnings - normal, 0)
    kpi = Table([[
        Paragraph(f"<b>Equipment</b><br/><font size=16 color='#17324D'>{len(result)}</font>", body),
        Paragraph(f"<b>Alarm</b><br/><font size=16 color='#E84A5F'>{alarms}</font>", body),
        Paragraph(f"<b>Warning</b><br/><font size=16 color='#FF8A4C'>{warnings}</font>", body),
        Paragraph(f"<b>Normal</b><br/><font size=16 color='#4CCB88'>{normal}</font>", body),
    ]], colWidths=[62*mm,62*mm,62*mm,62*mm])
    kpi.setStyle(TableStyle([
        ("ALIGN",(0,0),(-1,-1),"LEFT"),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    story.append(kpi)
    story.append(Spacer(1,2*mm))

    global _CURRENT_RESULT_FOR_PDF
    _CURRENT_RESULT_FOR_PDF = result
    chart_labels = Table([[
        Paragraph("<b>Status Distribution</b>", h2),
        Paragraph("<b>Status Count</b>", h2),
    ]], colWidths=[112*mm,138*mm])
    chart_labels.setStyle(TableStyle([
        ("LEFTPADDING",(0,0),(-1,-1),0), ("RIGHTPADDING",(0,0),(-1,-1),0),
        ("TOPPADDING",(0,0),(-1,-1),0), ("BOTTOMPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(chart_labels)
    charts = Table([[pdf_status_chart(result), pdf_status_count_chart(alarm_df)]],
                   colWidths=[112*mm,138*mm])
    charts.setStyle(TableStyle([
        ("BOX",(0,0),(-1,-1),0.35,colors.HexColor("#D0D7DE")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))
    story.append(charts)

    story.append(Paragraph("Priority Overview", h2))
    priority_rows = [["Rank","Equipment","Status","Continuous","Exceeded By","Peak °C"]]
    for rank, (_, r) in enumerate(alarm_df.head(5).iterrows(), 1):
        priority_rows.append([
            str(rank), Paragraph(str(r["Equipment"]), small),
            "URGENT >24h" if r["Longest Continuous"] > pd.Timedelta(hours=24) else "REVIEW",
            format_duration(r["Longest Continuous"]),
            format_duration(r["Exceeded By"]),
            f"{r['Peak During Excursion °C']:.1f}",
        ])
    pt = Table(priority_rows, colWidths=[13*mm,90*mm,35*mm,30*mm,30*mm,20*mm], repeatRows=1)
    pt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324D")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),7.2),
        ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#C5CCD3")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(0,1),(0,-1),"CENTER"),("ALIGN",(2,1),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))
    for i, (_, r) in enumerate(alarm_df.head(5).iterrows(), 1):
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
    story.append(Paragraph(
        "Longest Continuous uses consecutive samples at or above the applicable threshold; sampling gaps are not treated as continuous. "
        "Exceeded By is the portion beyond the applicable PPHG delay.",
        note
    ))

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

    headers = ["Equipment","Cat.","Min °C","Avg °C","Max °C","Start","End",
               "Continuous","Exceeded","Threshold Events","Status"]
    rows = [headers]
    for _, r in result.iterrows():
        start = r["Longest Start"].strftime("%d-%m %H:%M") if pd.notna(r["Longest Start"]) else "—"
        end = r["Longest End"].strftime("%d-%m %H:%M") if pd.notna(r["Longest End"]) else "—"
        rows.append([
            Paragraph(str(r["Equipment"]), tiny), str(r["Category"]),
            f"{r['Min °C']:.1f}", f"{r['Average °C']:.1f}", f"{r['Max °C']:.1f}",
            start, end, format_duration(r["Longest Continuous"]),
            format_duration(r["Exceeded By"]) if r["Exceeded By"] > pd.Timedelta(0) else "—",
            str(int(r["Threshold Events"])),
            "N/A" if r["Category"]=="Other" else str(r["Status"])
        ])
    at = Table(rows, colWidths=[51*mm,18*mm,13*mm,13*mm,13*mm,24*mm,24*mm,
                                26*mm,24*mm,18*mm,20*mm], repeatRows=1)
    cmd = [
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324D")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,0),6.6),("FONTSIZE",(0,1),(-1,-1),6.5),
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
    priority = [["Rank","Equipment","Category","Continuous","Exceeded By","Peak °C","Threshold Events"]]
    for rank, (_, r) in enumerate(alarm_df.iterrows(), 1):
        priority.append([
            str(rank), Paragraph(str(r["Equipment"]), small), str(r["Category"]),
            format_duration(r["Longest Continuous"]),
            format_duration(r["Exceeded By"]),
            f"{r['Peak During Excursion °C']:.1f}°C",
            str(int(r["Threshold Events"]))
        ])
    pt2 = Table(priority, colWidths=[13*mm,82*mm,24*mm,34*mm,34*mm,25*mm,25*mm], repeatRows=1)
    st2 = [
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324D")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),7.0),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#C5CCD3")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ALIGN",(0,1),(0,-1),"CENTER"),
        ("ALIGN",(2,1),(-1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),3.2),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]
    for i, (_, r) in enumerate(alarm_df.iterrows(),1):
        if r["Longest Continuous"] > pd.Timedelta(hours=24):
            st2.append(("BACKGROUND",(0,i),(-1,i),colors.HexColor("#FFF0F0")))
    pt2.setStyle(TableStyle(st2))
    story.append(pt2)
    story.append(Spacer(1,3*mm))
    story.append(Spacer(1,3*mm))
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
        "This section highlights recurrence frequency, not equipment-failure count or root cause.",
        note
    ))
    if repeat.empty:
        story.append(Paragraph("No repeated threshold events were detected.", body))
    else:
        rd = [["Equipment", "Category", "Threshold Events", "Longest Event", "Status"]]
        for _, r in repeat.iterrows():
            rd.append([
                Paragraph(str(r["Equipment"]), tiny),
                str(r["Category"]),
                str(int(r["Threshold Events"])),
                format_duration(r["Longest Continuous"]),
                "URGENT >24h" if r["Longest Continuous"] > pd.Timedelta(hours=24)
                else str(r["Status"])
            ])
        rt = Table(rd, colWidths=[88*mm,25*mm,34*mm,42*mm,40*mm], repeatRows=1)
        rt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EAF2F8")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#17324D")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#C5CCD3")),
            ("FONTSIZE",(0,0),(-1,-1),7),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("ALIGN",(1,1),(-1,-1),"CENTER"),
            ("TOPPADDING",(0,0),(-1,-1),3.5),
            ("BOTTOMPADDING",(0,0),(-1,-1),3.5),
        ]))
        for i, (_, r) in enumerate(repeat.iterrows(),1):
            if r["Longest Continuous"] > pd.Timedelta(hours=24):
                rt.setStyle(TableStyle([
                    ("BACKGROUND",(0,i),(-1,i),colors.HexColor("#FFF0F0"))
                ]))
        story.append(rt)
        story.append(Spacer(1,3*mm))
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



uploaded = st.file_uploader("Upload Testo CSV", type=["csv"])

if uploaded:
    try:
        data, raw = parse_testo(uploaded)
        result, median_interval = analyze(data)

        equipment_count = result["Equipment"].nunique()
        st.success(
            f"Testo CSV berhasil dibaca: {equipment_count} equipment dianalisis. "
            f"Sampling interval terdeteksi ≈ {round(median_interval.total_seconds()/60):g} menit."
        )

        other_count = int((result["Category"] == "Other").sum())
        if other_count:
            st.info(
                f"{other_count} measurement dikategorikan Other karena belum aman "
                f"diterapkan rule PPHG Chiller/Freezer. Setelah naming Testo dibuat unik, "
                f"mapping kategori bisa disesuaikan."
            )

        alarms = int((result["Status"] == "ALARM").sum())
        warnings = int((result["Status"] == "WARNING").sum())
        normal = int((result["Status"] == "NORMAL").sum())
        other_status = max(equipment_count - alarms - warnings - normal, 0)

        # ------------------------------------------------------------
        # Executive Summary
        # ------------------------------------------------------------
        st.subheader("Executive Summary")
        st.caption(
            "Overview of equipment status based on PPHG temperature threshold-event analysis."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Equipment", equipment_count)
        c2.metric("🔴 Alarm", alarms)
        c3.metric("🟠 Warning", warnings)
        c4.metric("🟢 Normal", normal)

        chart_left, chart_right = st.columns(2)

        # Status distribution donut.
        status_df = pd.DataFrame({
            "Status": ["Alarm", "Warning", "Normal", "Other / N/A"],
            "Count": [alarms, warnings, normal, other_status],
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
                     "Normal": "#4CCB88", "Other / N/A": "#9AA3AD"}[s]
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
        st.markdown("**Threshold Recurrence**")
        a1, a2 = st.columns(2)
        a1.metric("Units with Repeated Events", len(repeated_df))
        a2.metric("Total Threshold Events", int(result["Threshold Events"].sum()))

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

        if other_status:
            findings.append(
                f"**{other_status} measurement point(s)** are not classified as "
                f"Chiller/Freezer and should be mapped after Testo naming is made unique."
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
        category_options = ["All"] + sorted(
            result["Category"].dropna().unique().tolist()
        )
        status_options = ["All", "ALARM", "WARNING", "NORMAL", "SINGLE POINT", "N/A"]

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

        filtered_result = result.copy()
        if selected_category != "All":
            filtered_result = filtered_result[
                filtered_result["Category"] == selected_category
            ]
        if selected_status != "All":
            if selected_status == "N/A":
                filtered_result = filtered_result[
                    filtered_result["Category"] == "Other"
                ]
            else:
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
