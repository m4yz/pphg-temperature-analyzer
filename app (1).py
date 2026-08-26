import streamlit as st
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

    # A missing/gapped sample must not be treated as continuous excursion.
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
    """Calculate continuous excursions, recurrence count and recovery intervals."""
    if category not in RULES:
        return {
            "duration": pd.Timedelta(0), "start": pd.NaT, "end": pd.NaT,
            "peak": pd.NaT, "status": "N/A", "excursion_count": 0,
            "avg_excursion": pd.Timedelta(0), "avg_recovery": pd.NaT,
        }

    rule = RULES[category]
    g = g.sort_values("Timestamp").copy()
    g["above"] = g["Temperature"] >= rule["limit"]

    max_gap = median_interval * 1.5 if median_interval > pd.Timedelta(0) else pd.Timedelta(minutes=10)
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
            "peak": pd.NaT, "status": "NORMAL", "excursion_count": 0,
            "avg_excursion": pd.Timedelta(0), "avg_recovery": pd.NaT,
        }

    best = max(runs, key=lambda x: x["duration"])
    recovery = [
        nxt["start"] - prev["end"]
        for prev, nxt in zip(runs, runs[1:])
        if nxt["start"] - prev["end"] > pd.Timedelta(0)
    ]
    avg_recovery = (
        pd.Series(recovery, dtype="timedelta64[ns]").mean()
        if recovery else pd.NaT
    )

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
        "excursion_count": len(runs),
        "avg_excursion": pd.Series(
            [r["duration"] for r in runs], dtype="timedelta64[ns]"
        ).mean(),
        "avg_recovery": avg_recovery,
    }


def recommendation_for_row(row):
    category = row["Category"]
    peak = row["Peak During Excursion °C"]
    duration = row["Longest Continuous"]
    repeated = int(row.get("Excursion Count", 0))

    if category == "Freezer":
        if pd.notna(peak) and peak >= 0:
            text = "Check refrigeration performance, door/door-seal condition, defrost cycle, loading and product obstruction."
        else:
            text = "Review door opening/loading, refrigeration performance, condenser/coil condition and defrost operation."
    elif category == "Chiller":
        text = "Review door opening/loading, ambient exposure, condenser/coil condition, airflow and temperature-control performance."
    else:
        text = "Map the measurement point to the correct equipment category."

    if repeated >= 3:
        text += " Repeated excursions also warrant review of operating practices and recurrence pattern."
    if duration > pd.Timedelta(hours=24):
        text += " Prioritize this unit for urgent operational review due to the prolonged excursion."
    return text


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
            "Excursion Count": stats["excursion_count"],
            "Avg Excursion": stats["avg_excursion"],
            "Avg Recovery Interval": stats["avg_recovery"],
        })

    out = pd.DataFrame(rows)
    order = {"ALARM": 0, "WARNING": 1, "NORMAL": 2, "N/A": 3}
    out["_order"] = out["Status"].map(order).fillna(9)
    out = out.sort_values(
        ["_order", "Longest Continuous"],
        ascending=[True, False]
    ).drop(columns="_order")
    return out, median_interval



def pdf_status_chart(result):
    counts = [
        int((result["Status"] == "ALARM").sum()),
        int((result["Status"] == "WARNING").sum()),
        int((result["Status"] == "NORMAL").sum()),
        int((result["Category"] == "Other").sum()) + int((result["Status"] == "SINGLE POINT").sum()),
    ]
    labels = ["Alarm", "Warning", "Normal", "Other / N/A"]
    fills = [
        colors.HexColor("#E84A5F"),
        colors.HexColor("#FF8A4C"),
        colors.HexColor("#4CCB88"),
        colors.HexColor("#9AA3AD"),
    ]

    d = Drawing(320, 145)
    pie = Pie()
    pie.x, pie.y, pie.width, pie.height = 55, 8, 120, 120
    pie.data = counts
    pie.labels = ["" for _ in counts]
    for i, fill in enumerate(fills):
        pie.slices[i].fillColor = fill
    pie.slices.strokeWidth = 0.5
    pie.slices.strokeColor = colors.white
    d.add(pie)

    legend = Legend()
    legend.x, legend.y = 190, 105
    legend.fontName, legend.fontSize = "Helvetica", 8
    legend.colorNamePairs = [
        (fills[i], f"{labels[i]}  {counts[i]}")
        for i in range(4) if counts[i] > 0
    ]
    d.add(legend)
    return d


def pdf_top5_chart(alarm_df):
    top = alarm_df.head(5).iloc[::-1]
    d = Drawing(430, 165)
    if top.empty:
        return d

    chart = HorizontalBarChart()
    chart.x, chart.y, chart.width, chart.height = 120, 18, 285, 125
    chart.data = [[
        max(0, round(v.total_seconds() / 3600, 1))
        for v in top["Longest Continuous"]
    ]]
    chart.categoryAxis.categoryNames = [str(v)[:38] for v in top["Equipment"]]
    chart.valueAxis.valueMin = 0
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.labelTextFormat = "%.0fh"
    chart.categoryAxis.labels.fontSize = 6.5
    chart.bars[0].fillColor = colors.HexColor("#E84A5F")
    chart.bars[0].strokeColor = colors.HexColor("#E84A5F")
    chart.barWidth = 12
    d.add(chart)
    return d


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


def _action_short(row):
    if row["Category"] == "Freezer":
        action = "Refrigeration performance; door/seal; defrost; loading; condenser/coil."
    elif row["Category"] == "Chiller":
        action = "Door/loading; ambient exposure; condenser/coil; airflow; temperature control."
    else:
        action = "Confirm equipment mapping before operational assessment."
    if int(row.get("Excursion Count", 0)) >= 3:
        action += " Review recurrence pattern."
    return action


def pdf_status_chart(result):
    counts = [
        int((result["Status"] == "ALARM").sum()),
        int((result["Status"] == "WARNING").sum()),
        int((result["Status"] == "NORMAL").sum()),
        int((result["Category"] == "Other").sum()),
        int((result["Status"] == "SINGLE POINT").sum()),
    ]
    labels = ["Alarm", "Warning", "Normal", "Other / N/A", "Single Point"]
    fills = [
        colors.HexColor("#E84A5F"),
        colors.HexColor("#FF9F43"),
        colors.HexColor("#49C78A"),
        colors.HexColor("#9AA3AD"),
        colors.HexColor("#F4C95D"),
    ]
    total = sum(counts)
    d = Drawing(320, 150)
    pie = Pie()
    pie.x, pie.y, pie.width, pie.height = 55, 8, 120, 120
    pie.data = counts
    pie.labels = ["" for _ in counts]
    for i, fill in enumerate(fills):
        pie.slices[i].fillColor = fill
        pie.slices[i].strokeColor = colors.white
        pie.slices[i].strokeWidth = 0.5
    d.add(pie)

    legend = Legend()
    legend.x, legend.y = 190, 112
    legend.fontName, legend.fontSize = "Helvetica", 7.5
    legend.colorNamePairs = [
        (fills[i], f"{labels[i]}  {counts[i]}")
        for i in range(len(counts)) if counts[i] > 0
    ]
    d.add(legend)
    return d


def pdf_top5_chart(alarm_df):
    from reportlab.graphics.shapes import String, Rect, Line
    top = alarm_df.head(5)
    d = Drawing(430, 150)
    if top.empty:
        return d

    values = [max(0, round(v.total_seconds()/3600, 1)) for v in top["Longest Continuous"]]
    maxv = max(values) if values else 1
    left, right = 155, 410
    row_h = 25
    for idx, ((_, row), value) in enumerate(zip(top.iterrows(), values)):
        y = 118 - idx * row_h
        name = str(row["Equipment"])
        if len(name) > 29:
            name = name[:27] + "…"
        d.add(String(148, y+3, name, fontName="Helvetica", fontSize=6.8,
                     textAnchor="end", fillColor=colors.HexColor("#263746")))
        width = (right-left) * value / maxv if maxv else 0
        d.add(Rect(left, y-3, width, 12, fillColor=colors.HexColor("#E84A5F"),
                   strokeColor=None))
        d.add(String(min(left+width+5, right-2), y, f"{value:g}h",
                     fontName="Helvetica-Bold", fontSize=7,
                     fillColor=colors.HexColor("#263746")))
    d.add(Line(left, 8, right, 8, strokeColor=colors.HexColor("#B9C2CC"), strokeWidth=0.6))
    d.add(String(left, 0, "Longest Continuous excursion (hours)",
                 fontName="Helvetica", fontSize=6.8,
                 fillColor=colors.HexColor("#667481")))
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
        ["Longest Continuous", "Excursion Count"], ascending=[False, False]
    ).copy()
    single_df = result[result["Status"] == "SINGLE POINT"].copy()
    repeat = result[result["Excursion Count"] >= 2].sort_values(
        ["Excursion Count", "Longest Continuous"], ascending=[False, False]
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

    kpi = Table([[
        Paragraph(f"<b>ALARM</b><br/><font size=16 color='#B00020'>{alarms}</font>", body),
        Paragraph(f"<b>WARNING</b><br/><font size=16 color='#A35B00'>{warnings}</font>", body),
        Paragraph(f"<b>NORMAL</b><br/><font size=16 color='#087443'>{normal}</font>", body),
        Paragraph(f"<b>OTHER / N/A</b><br/><font size=16 color='#59636D'>{other}</font>", body),
        Paragraph(f"<b>SINGLE POINT</b><br/><font size=16 color='#8A6B00'>{single}</font>", body),
    ]], colWidths=[48*mm]*5)
    kpi.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,0),colors.HexColor("#FFE3E3")),
        ("BACKGROUND",(1,0),(1,0),colors.HexColor("#FFF0CC")),
        ("BACKGROUND",(2,0),(2,0),colors.HexColor("#D9F5E5")),
        ("BACKGROUND",(3,0),(3,0),colors.HexColor("#EEF1F4")),
        ("BACKGROUND",(4,0),(4,0),colors.HexColor("#FFF7D6")),
        ("BOX",(0,0),(-1,-1),0.35,colors.HexColor("#D0D7DE")),
        ("INNERGRID",(0,0),(-1,-1),0.35,colors.white),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(kpi)
    story.append(Spacer(1,2*mm))

    charts = Table([[pdf_status_chart(result), pdf_top5_chart(alarm_df)]],
                   colWidths=[92*mm,158*mm])
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
        ["Chiller", "≥ 6°C", "2 hours", "Continuous excursion ≥2h → ALARM"],
        ["Freezer", "≥ -15°C", "4 hours", "Continuous excursion ≥4h → ALARM"],
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
        "Longest Continuous uses consecutive samples above the applicable threshold; sampling gaps are not treated as continuous. "
        "Exceeded By is the portion beyond the applicable PPHG delay.",
        note
    ))

    # PAGE 2 — Equipment Analysis
    story.append(PageBreak())
    story.append(Paragraph("2. Equipment Analysis", title))
    story.append(Paragraph(
        f"<b>{len(result)} equipment</b> • Sorted by status and longest continuous excursion. "
        "Status colors are applied only to the final column for quick scanning.",
        note
    ))

    headers = ["Equipment","Cat.","Min °C","Avg °C","Max °C","Start","End",
               "Continuous","Exceeded","Excursions","Status"]
    rows = [headers]
    for _, r in result.iterrows():
        start = r["Longest Start"].strftime("%d-%m %H:%M") if pd.notna(r["Longest Start"]) else "—"
        end = r["Longest End"].strftime("%d-%m %H:%M") if pd.notna(r["Longest End"]) else "—"
        rows.append([
            Paragraph(str(r["Equipment"]), tiny), str(r["Category"]),
            f"{r['Min °C']:.1f}", f"{r['Average °C']:.1f}", f"{r['Max °C']:.1f}",
            start, end, format_duration(r["Longest Continuous"]),
            format_duration(r["Exceeded By"]) if r["Exceeded By"] > pd.Timedelta(0) else "—",
            str(int(r["Excursion Count"])),
            "N/A" if r["Category"]=="Other" else str(r["Status"])
        ])
    at = Table(rows, colWidths=[51*mm,18*mm,13*mm,13*mm,13*mm,24*mm,24*mm,
                                26*mm,24*mm,18*mm,20*mm], repeatRows=1)
    cmd = [
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324D")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,0),6.8),("FONTSIZE",(0,1),(-1,-1),6.8),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#C5CCD3")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ALIGN",(2,1),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),3.2),("BOTTOMPADDING",(0,0),(-1,-1),3.2),
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
    story.append(Spacer(1,2*mm))
    story.append(Paragraph(
        "<b>Reading guide:</b> Continuous = longest threshold excursion. Exceeded = time beyond the PPHG delay. "
        "Excursions = number of distinct threshold excursions detected for that equipment.",
        note
    ))

    # PAGE 3 — Management actions
    story.append(PageBreak())
    story.append(Paragraph("3. Management Analysis & Recommended Actions", title))
    story.append(Paragraph(
        f"<b>{alarms} ALARM equipment</b> identified. "
        f"<b>{urgent}</b> have continuous excursions &gt;24h and are classified <b>REVIEW URGENTLY</b>.",
        callout
    ))
    priority = [["Priority","Equipment","Cat.","Continuous","Exceeded","Peak","Action / review focus"]]
    for rank, (_, r) in enumerate(alarm_df.iterrows(), 1):
        urgent_tag = "URGENT >24h" if r["Longest Continuous"] > pd.Timedelta(hours=24) else "Review"
        priority.append([
            str(rank), Paragraph(str(r["Equipment"]), tiny), str(r["Category"]),
            format_duration(r["Longest Continuous"]),
            format_duration(r["Exceeded By"]),
            f"{r['Peak During Excursion °C']:.1f}°C",
            Paragraph(f"<b>{urgent_tag}</b> — {_action_short(r)}", tiny)
        ])
    pt2 = Table(priority, colWidths=[13*mm,57*mm,18*mm,28*mm,27*mm,18*mm,87*mm], repeatRows=1)
    st2 = [
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324D")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),6.7),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#C5CCD3")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ALIGN",(0,1),(0,-1),"CENTER"),
        ("ALIGN",(2,1),(5,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]
    for i, (_, r) in enumerate(alarm_df.iterrows(),1):
        if r["Longest Continuous"] > pd.Timedelta(hours=24):
            st2.append(("BACKGROUND",(0,i),(-1,i),colors.HexColor("#FFF0F0")))
    pt2.setStyle(TableStyle(st2))
    story.append(pt2)
    story.append(Spacer(1,3*mm))
    story.append(Paragraph(
        "<b>Management interpretation:</b> These recommendations are operational review prompts, not root-cause diagnoses. "
        "Confirm causes against maintenance history, loading/door records, ambient conditions and equipment operating records.",
        note
    ))

    # PAGE 4 — Warning + single point
    story.append(PageBreak())
    story.append(Paragraph("4. Warning & Single-Point Review", title))
    if warnings:
        story.append(Paragraph("WARNING — Monitor & Follow Up", h2))
        wd = [["Equipment","Category","Continuous","Excursions","Peak °C","Management focus"]]
        for _, r in warning_df.iterrows():
            wd.append([
                Paragraph(str(r["Equipment"]), tiny), str(r["Category"]),
                format_duration(r["Longest Continuous"]), str(int(r["Excursion Count"])),
                f"{r['Peak During Excursion °C']:.1f}",
                Paragraph(_action_short(r), tiny)
            ])
        wt = Table(wd, colWidths=[58*mm,22*mm,28*mm,22*mm,22*mm,88*mm], repeatRows=1)
        wt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#FFF0CC")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#6B4A00")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#C5CCD3")),
            ("FONTSIZE",(0,0),(-1,-1),6.8),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("ALIGN",(1,1),(4,-1),"CENTER"),
            ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ]))
        story.append(wt)
        story.append(Paragraph(
            "Warning units did not reach the applicable alarm duration. Repeated excursions or durations approaching the alarm delay should receive follow-up.",
            note
        ))

    if not single_df.empty:
        story.append(Paragraph("Single-Point Excursions", h2))
        story.append(Paragraph(
            "A threshold excursion was observed, but no elapsed duration can be established from the available point(s). "
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

    # PAGE 5 — Recurrence
    story.append(PageBreak())
    story.append(Paragraph("5. Repeated Excursions & Recovery Interval", title))
    story.append(Paragraph(
        f"<b>{len(repeat)}</b> equipment recorded multiple distinct threshold excursions. "
        "This section highlights recurrence frequency rather than alarm severity.",
        note
    ))
    if repeat.empty:
        story.append(Paragraph("No repeated threshold excursions were detected.", body))
    else:
        rd = [["Equipment","Category","Excursions","Avg Excursion","Avg Recovery Interval"]]
        for _, r in repeat.iterrows():
            rd.append([
                Paragraph(str(r["Equipment"]), tiny), str(r["Category"]),
                str(int(r["Excursion Count"])), format_duration(r["Avg Excursion"]),
                format_duration(r["Avg Recovery Interval"]) if pd.notna(r["Avg Recovery Interval"]) else "—"
            ])
        rt = Table(rd, colWidths=[92*mm,25*mm,25*mm,38*mm,50*mm], repeatRows=1)
        rt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EAF2F8")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#17324D")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#C5CCD3")),
            ("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("ALIGN",(1,1),(-1,-1),"CENTER"),
            ("TOPPADDING",(0,0),(-1,-1),3.5),("BOTTOMPADDING",(0,0),(-1,-1),3.5),
        ]))
        for i, (_, r) in enumerate(repeat.iterrows(),1):
            if int(r["Excursion Count"]) >= 50:
                rt.setStyle(TableStyle([("BACKGROUND",(0,i),(-1,i),colors.HexColor("#FFF5F5"))]))
        story.append(rt)
        story.append(Spacer(1,3*mm))
        story.append(Paragraph(
            "<b>Recovery interval</b> is the elapsed time between the end of one detected threshold excursion and the start of the next. "
            "It is a recurrence indicator, not a direct compressor/refrigeration recovery measurement.",
            note
        ))
    story.append(Spacer(1,5*mm))
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
            "Overview of equipment status based on PPHG temperature excursion analysis."
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

        # Phase 1 enhancements: repeated excursions and recovery interval.
        repeated_df = result[result["Excursion Count"] >= 2].copy()
        st.markdown("**Additional Analysis**")
        a1, a2 = st.columns(2)
        a1.metric("Repeated Excursion Units", len(repeated_df))
        a2.metric("Total Excursions", int(result["Excursion Count"].sum()))

        if not repeated_df.empty:
            top_repeat = repeated_df.sort_values(
                ["Excursion Count", "Longest Continuous"],
                ascending=[False, False]
            ).iloc[0]
            st.info(
                f"**Repeated excursions:** {top_repeat['Equipment']} recorded "
                f"**{int(top_repeat['Excursion Count'])} distinct excursions**. "
                f"Average excursion: **{format_duration(top_repeat['Avg Excursion'])}**. "
                f"Average interval between excursions: **"
                f"{format_duration(top_repeat['Avg Recovery Interval']) if pd.notna(top_repeat['Avg Recovery Interval']) else '—'}**."
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
                    f"**{urgent} equipment** have a continuous excursion exceeding "
                    f"24 hours and should receive priority operational review."
                )

            top = alarm_df.iloc[0]
            findings.append(
                f"Longest current excursion: **{top['Equipment']}** — "
                f"{format_duration(top['Longest Continuous'])}, "
                f"peak **{top['Peak During Excursion °C']:.1f}°C**."
            )

        if warnings:
            findings.append(
                f"**{warnings} equipment** recorded excursions below the PPHG "
                f"alarm-duration threshold and should be monitored for recurrence."
            )

        if other_status:
            findings.append(
                f"**{other_status} measurement point(s)** are not classified as "
                f"Chiller/Freezer and should be mapped after Testo naming is made unique."
            )

        if not findings:
            findings.append("No PPHG Chiller/Freezer alarm-duration excursions were identified.")

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

            **Longest Continuous** = longest continuous excursion above the applicable
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
            "Longest Continuous", "Exceeded By", "Excursion Count", "Status"
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
            "Excursion Start",
            selected_result["Longest Start"].strftime("%d-%m-%Y %H:%M")
            if pd.notna(selected_result["Longest Start"]) else "—"
        )
        d3.metric(
            "Excursion End",
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
