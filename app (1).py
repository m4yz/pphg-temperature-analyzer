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
from reportlab.graphics.shapes import Drawing, Rect, String
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


@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
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
            limit_text = f"≥{float(rule['limit']):g}°C / {float(rule['delay'].total_seconds()/3600):g}h"
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
    """Clean dashboard-matched status distribution donut for PDF."""
    alarms = int((result["Status"] == "ALARM").sum())
    warnings = int((result["Status"] == "WARNING").sum())
    normal = int((result["Status"] == "NORMAL").sum())
    other_status = max(len(result) - alarms - warnings - normal, 0)

    labels = ["Alarm", "Warning", "Normal", "Other / N/A"]
    counts = [alarms, warnings, normal, other_status]
    fills = [
        colors.HexColor("#E84A5F"),
        colors.HexColor("#FF8A4C"),
        colors.HexColor("#4CCB88"),
        colors.HexColor("#9AA3AD"),
    ]

    from reportlab.graphics.shapes import String, Rect
    d = Drawing(300, 135)

    # Donut only — no center total, so it cannot collide with the chart.
    pie = Pie()
    pie.x, pie.y, pie.width, pie.height = 12, 8, 112, 112
    pie.data = counts
    pie.labels = ["" for _ in counts]
    pie.slices[0].fillColor = fills[0]
    for i, fill in enumerate(fills):
        pie.slices[i].fillColor = fill
        pie.slices[i].strokeColor = colors.white
        pie.slices[i].strokeWidth = 0.7
    d.add(pie)

    # Manual legend: fixed positions prevent ReportLab Legend from
    # overflowing into the adjacent Status Count chart.
    y = 103
    for label, count, fill in zip(labels, counts, fills):
        if count <= 0:
            continue
        d.add(Rect(150, y-3, 9, 9, fillColor=fill, strokeColor=colors.HexColor("#6B7785"), strokeWidth=0.35))
        d.add(String(166, y, f"{label}  {count}", fontName="Helvetica", fontSize=8,
                   fillColor=colors.HexColor("#334155")))
        y -= 22
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
    d = Drawing(405, 135)
    left, right = 44, 390
    bottom, top = 20, 110
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

    return d


def _pdf_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(16*mm, 9*mm, "PPHG Temperature Analyzer • Analytical screening report")
    canvas.drawRightString(194*mm, 9*mm, f"Page {doc.page}")
    canvas.restoreState()


def _pdf_short_name(name, n=40):
    s = str(name)
    return s if len(s) <= n else s[:n-1] + "…"



def _interval_minutes(value):
    """Return a numeric sampling interval in minutes, regardless of input type."""
    try:
        return float(pd.to_timedelta(value).total_seconds() / 60.0)
    except Exception:
        return float(value)


def build_pdf_report(result, data, median_interval, raw=None):
    """Clean 5-page management report. Dashboard and PDF intentionally use different density."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=16*mm,
        rightMargin=16*mm,
        topMargin=15*mm,
        bottomMargin=14*mm,
        title="PPHG Temperature Analysis Report",
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "V10Title", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=18, leading=21, textColor=colors.HexColor("#183B56"),
        spaceAfter=6
    )
    section = ParagraphStyle(
        "V10Section", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=12.5, leading=15, textColor=colors.HexColor("#183B56"),
        spaceBefore=2, spaceAfter=6
    )
    body = ParagraphStyle(
        "V10Body", parent=styles["BodyText"], fontSize=8.2, leading=10.2,
        textColor=colors.HexColor("#344054")
    )
    note = ParagraphStyle(
        "V10Note", parent=body, fontSize=7.7, leading=9.8,
        textColor=colors.HexColor("#475467")
    )
    tiny = ParagraphStyle(
        "V10Tiny", parent=body, fontSize=6.7, leading=8.1
    )

    total = len(result)
    counts = result["Status"].value_counts().to_dict()
    alarm = int(counts.get("ALARM", 0))
    warning = int(counts.get("WARNING", 0))
    normal = int(counts.get("NORMAL", 0))
    other = total - alarm - warning - normal

    def td_text(v):
        if pd.isna(v):
            return "—"
        mins = int(round(pd.to_timedelta(v).total_seconds() / 60))
        d, rem = divmod(mins, 1440)
        h, m = divmod(rem, 60)
        if d:
            return f"{d}d {h}h {m}m"
        if h:
            return f"{h}h {m}m"
        return f"{m}m"

    def dt_text(v):
        if pd.isna(v):
            return "—"
        return pd.to_datetime(v).strftime("%d-%m-%Y %H:%M")

    def status_bg(status):
        return {
            "ALARM": colors.HexColor("#FDE2E5"),
            "WARNING": colors.HexColor("#FFF0D9"),
            "NORMAL": colors.HexColor("#DDF5E9"),
        }.get(str(status), colors.HexColor("#EEF2F6"))

    story = []

    # ==================== PAGE 1 ====================
    story.append(Paragraph("1. Executive Summary", title))

    period = ""
    if "Timestamp" in data.columns and not data["Timestamp"].dropna().empty:
        tmin = pd.to_datetime(data["Timestamp"]).min()
        tmax = pd.to_datetime(data["Timestamp"]).max()
        period = f" • Analysis period: {tmin:%d %b %Y %H:%M} – {tmax:%d %b %Y %H:%M}"

    callout = Table(
        [[Paragraph(
            f"<b>Overall Assessment: {'ATTENTION REQUIRED' if alarm else 'NO ALARM IDENTIFIED'}</b>"
            f"{period} • Sampling interval: approximately {_interval_minutes(median_interval):g} min",
            body
        )]],
        colWidths=[178*mm],
        style=TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#EEF6FB")),
            ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#B8D2E5")),
            ("LEFTPADDING",(0,0),(-1,-1),5),
            ("RIGHTPADDING",(0,0),(-1,-1),5),
            ("TOPPADDING",(0,0),(-1,-1),5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ])
    )
    story.append(callout)
    story.append(Spacer(1, 7))

    # Four KPI cards — deliberately separated from the chart area.
    kpis = [
        ("EQUIPMENT", total, "#F2F6FA", "#344054"),
        ("ALARM", alarm, "#FDE3E6", "#E94B63"),
        ("WARNING", warning, "#FFF0D9", "#D97706"),
        ("NORMAL", normal, "#DDF5E9", "#12A36A"),
    ]
    kdata = [[
        Paragraph(f"<b>{kpis[0][0]}</b><br/><font size=17 color='{kpis[0][3]}'>{kpis[0][1]}</font>", body),
        Paragraph(f"<b>{kpis[1][0]}</b><br/><font size=17 color='{kpis[1][3]}'>{kpis[1][1]}</font>", body),
        Paragraph(f"<b>{kpis[2][0]}</b><br/><font size=17 color='{kpis[2][3]}'>{kpis[2][1]}</font>", body),
        Paragraph(f"<b>{kpis[3][0]}</b><br/><font size=17 color='{kpis[3][3]}'>{kpis[3][1]}</font>", body),
    ]]
    kt = Table(kdata, colWidths=[44.5*mm]*4, rowHeights=[15*mm])
    kt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,0),colors.HexColor(kpis[0][2])),
        ("BACKGROUND",(1,0),(1,0),colors.HexColor(kpis[1][2])),
        ("BACKGROUND",(2,0),(2,0),colors.HexColor(kpis[2][2])),
        ("BACKGROUND",(3,0),(3,0),colors.HexColor(kpis[3][2])),
        ("BOX",(0,0),(-1,-1),0.35,colors.HexColor("#D0D5DD")),
        ("INNERGRID",(0,0),(-1,-1),0.35,colors.HexColor("#D0D5DD")),
        ("LEFTPADDING",(0,0),(-1,-1),6),
        ("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story.append(kt)
    story.append(Spacer(1, 10))

    # One visual only: status donut. No bar chart in PDF.
    story.append(Paragraph("Status Distribution", section))
    chart = Drawing(178*mm, 58*mm)
    pie = Pie()
    pie.x, pie.y, pie.width, pie.height = 24*mm, 3*mm, 48*mm, 48*mm
    pie.data = [alarm, warning, normal, other]
    pie.labels = ["", "", "", ""]
    pie.slices[0].fillColor = colors.HexColor("#E94B63")
    pie.slices[1].fillColor = colors.HexColor("#FF8A4C")
    pie.slices[2].fillColor = colors.HexColor("#42C88A")
    pie.slices[3].fillColor = colors.HexColor("#98A2B3")
    for s in pie.slices:
        s.strokeColor = colors.white
        s.strokeWidth = 1
    chart.add(pie)

    legend = [
        ("Alarm", alarm, "#E94B63"),
        ("Warning", warning, "#FF8A4C"),
        ("Normal", normal, "#42C88A"),
        ("Other / N/A", other, "#98A2B3"),
    ]
    for i, (label, value, color) in enumerate(legend):
        y = 44*mm - i*10*mm
        chart.add(Rect(82*mm, y, 4*mm, 4*mm, fillColor=colors.HexColor(color), strokeColor=None))
        chart.add(String(89*mm, y+0.8*mm, f"{label}: {value}", fontName="Helvetica", fontSize=8.5, fillColor=colors.HexColor("#344054")))
    chart.add(String(20*mm, 0, "Status distribution of analyzed equipment", fontName="Helvetica", fontSize=7.2, fillColor=colors.HexColor("#667085")))
    story.append(chart)
    story.append(Spacer(1, 6))

    priority = result[result["Status"] == "ALARM"].copy()
    priority["_sort"] = priority["Longest Continuous"].apply(lambda x: x.total_seconds() if pd.notna(x) else -1)
    priority = priority.sort_values("_sort", ascending=False).head(5)

    story.append(Paragraph("Priority Overview", section))
    p = [["Rank", "Equipment", "Continuous", "Exceeded By", "Peak °C"]]
    for i, (_, r) in enumerate(priority.iterrows(), 1):
        p.append([str(i), _pdf_short_name(r["Equipment"], 43), td_text(r["Longest Continuous"]),
                  td_text(r["Exceeded By"]), f"{r['Peak °C']:.1f}"])
    pt = Table(p, colWidths=[12*mm, 82*mm, 30*mm, 30*mm, 24*mm], repeatRows=1)
    pt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#183B56")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),7.5),
        ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#CBD5E1")),
        ("ALIGN",(0,0),(0,-1),"CENTER"),
        ("ALIGN",(2,1),(-1,-1),"CENTER"),
        ("BACKGROUND",(0,1),(-1,-1),colors.HexColor("#FFF3F4")),
        ("TOPPADDING",(0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    story.append(pt)
    story.append(Spacer(1, 7))

    urgent = result[
        (result["Status"] == "ALARM") &
        (result["Longest Continuous"].notna()) &
        (result["Longest Continuous"] > pd.Timedelta(hours=24))
    ]
    story.append(Paragraph(
        f"<b>Key finding:</b> {len(urgent)} equipment have continuous temperature excursions exceeding 24 hours. "
        "This duration is used as a priority operational flag and may indicate a potential sustained equipment performance issue; "
        "it is not a root-cause diagnosis.",
        note
    ))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "<b>PPHG criteria:</b> Chiller ≥6°C for 2 hours • Freezer ≥−15°C for 4 hours. "
        "Longest Continuous uses consecutive samples above the applicable threshold.",
        note
    ))

    # ==================== PAGE 2 ====================
    story.append(PageBreak())
    story.append(Paragraph("2. Equipment Analysis", title))
    story.append(Paragraph("Detailed screening results for all equipment included in the analysis.", body))
    story.append(Spacer(1, 7))

    cols = ["Equipment","Category","Min °C","Average °C","Max °C","Longest Start","Longest End","Longest Continuous","Exceeded By","Status"]
    ed = [cols]
    for _, r in result.iterrows():
        ed.append([
            _pdf_short_name(r["Equipment"], 31), str(r["Category"]),
            f"{r['Min °C']:.1f}", f"{r['Average °C']:.1f}", f"{r['Max °C']:.1f}",
            dt_text(r["Longest Start"]), dt_text(r["Longest End"]),
            td_text(r["Longest Continuous"]), td_text(r["Exceeded By"]), str(r["Status"])
        ])
    et = Table(ed, colWidths=[38*mm,19*mm,12*mm,14*mm,12*mm,22*mm,22*mm,25*mm,23*mm,18*mm], repeatRows=1)
    es = [
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#183B56")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),6.6),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CBD5E1")),
        ("ALIGN",(2,1),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),3.2),
        ("BOTTOMPADDING",(0,0),(-1,-1),3.2),
    ]
    for i, (_, r) in enumerate(result.iterrows(), 1):
        es.append(("BACKGROUND",(9,i),(9,i),{
            "ALARM": colors.HexColor("#FDE2E5"),
            "WARNING": colors.HexColor("#FFF0D9"),
            "NORMAL": colors.HexColor("#DDF5E9")
        }.get(str(r["Status"]), colors.HexColor("#EEF2F6"))))
        es.append(("FONTNAME",(9,i),(9,i),"Helvetica-Bold"))
    et.setStyle(TableStyle(es))
    story.append(et)

    # ==================== PAGE 3 ====================
    story.append(PageBreak())
    story.append(Paragraph("3. Alarm Analysis & Priority Review", title))
    alarms = result[result["Status"] == "ALARM"].copy()
    alarms["_sort"] = alarms["Longest Continuous"].apply(lambda x: x.total_seconds() if pd.notna(x) else -1)
    alarms = alarms.sort_values("_sort", ascending=False)

    story.append(Paragraph(
        f"<b>{alarm} ALARM equipment identified.</b> {len(urgent)} have continuous excursions &gt;24 hours and are flagged for priority operational review.",
        body
    ))
    story.append(Spacer(1, 7))

    ad = [["Rank","Equipment","Category","Continuous","Exceeded By","Peak °C"]]
    for i, (_, r) in enumerate(alarms.iterrows(), 1):
        ad.append([str(i), _pdf_short_name(r["Equipment"], 42), str(r["Category"]),
                   td_text(r["Longest Continuous"]), td_text(r["Exceeded By"]), f"{r['Peak °C']:.1f}"])
    at = Table(ad, colWidths=[13*mm,70*mm,22*mm,29*mm,29*mm,22*mm], repeatRows=1)
    at.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#183B56")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),7.4),
        ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#CBD5E1")),
        ("ALIGN",(0,0),(0,-1),"CENTER"),
        ("ALIGN",(2,1),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),4.5),
        ("BOTTOMPADDING",(0,0),(-1,-1),4.5),
    ]))
    story.append(at)
    story.append(Spacer(1, 9))
    story.append(Paragraph(
        "<b>Priority note:</b> Continuous excursions exceeding 24 hours are treated as a strong operational flag for possible sustained equipment performance issues. "
        "Review equipment condition against door/loading activity, ambient exposure, maintenance history and operating records. "
        "This analysis does not establish root cause.",
        note
    ))

    # ==================== PAGE 4 ====================
    story.append(PageBreak())
    story.append(Paragraph("4. Warning & Single-Point Review", title))

    warn = result[result["Status"] == "WARNING"].copy()
    story.append(Paragraph(
        f"<b>{len(warn)} WARNING equipment.</b> Excursions were detected, but the longest continuous duration did not exceed the applicable PPHG alarm delay.",
        body
    ))
    story.append(Spacer(1, 7))

    wd = [["Equipment","Category","Continuous","Peak °C","Threshold Events"]]
    for _, r in warn.iterrows():
        wd.append([_pdf_short_name(r["Equipment"], 47), str(r["Category"]), td_text(r["Longest Continuous"]),
                   f"{r['Peak °C']:.1f}", str(int(r.get("Threshold Events", 0)))])
    wt = Table(wd, colWidths=[72*mm,25*mm,32*mm,24*mm,25*mm], repeatRows=1)
    wt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#183B56")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),7.7),
        ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#CBD5E1")),
        ("ALIGN",(2,1),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),4.5),
        ("BOTTOMPADDING",(0,0),(-1,-1),4.5),
    ]))
    story.append(wt)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Warning note:</b> Repeated short excursions may still warrant operational review even when the PPHG alarm delay is not exceeded.",
        note
    ))
    story.append(Spacer(1, 14))

    other = result[result["Status"].isin(["OTHER / N/A","SINGLE POINT"])].copy()
    story.append(Paragraph(f"<b>{len(other)} Other / N/A or Single-Point equipment.</b>", body))
    story.append(Spacer(1, 7))
    od = [["Equipment","Category","Status","Peak °C","Threshold Events"]]
    for _, r in other.iterrows():
        od.append([_pdf_short_name(r["Equipment"], 47), str(r["Category"]), str(r["Status"]),
                   f"{r['Peak °C']:.1f}", str(int(r.get("Threshold Events", 0)))])
    ot = Table(od, colWidths=[72*mm,25*mm,32*mm,24*mm,25*mm], repeatRows=1)
    ot.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EEF2F6")),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),7.7),
        ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#CBD5E1")),
        ("ALIGN",(2,1),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),4.5),
        ("BOTTOMPADDING",(0,0),(-1,-1),4.5),
    ]))
    story.append(ot)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Single-point note:</b> A single threshold-crossing observation is not treated as a sustained alarm event. "
        "Other/N/A equipment is excluded from Chiller/Freezer alarm classification until valid category mapping is available.",
        note
    ))

    # ==================== PAGE 5 ====================
    story.append(PageBreak())
    story.append(Paragraph("5. Threshold Recurrence", title))
    story.append(Paragraph(
        "Threshold Events represent distinct periods where measured temperature crossed the applicable PPHG threshold. "
        "They are recurrence indicators, not counts of equipment failures.",
        body
    ))
    story.append(Spacer(1, 7))

    rec = result.copy()
    rec["_events"] = pd.to_numeric(rec.get("Threshold Events", 0), errors="coerce").fillna(0)
    rec = rec[rec["_events"] > 0].sort_values(["_events","Longest Continuous"], ascending=[False,False])

    rd = [["Equipment","Category","Status","Threshold Events","Longest Continuous"]]
    for _, r in rec.iterrows():
        rd.append([_pdf_short_name(r["Equipment"], 47), str(r["Category"]), str(r["Status"]),
                   str(int(r["_events"])), td_text(r["Longest Continuous"])])
    rt = Table(rd, colWidths=[72*mm,25*mm,28*mm,30*mm,31*mm], repeatRows=1)
    rt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#183B56")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),7.7),
        ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#CBD5E1")),
        ("ALIGN",(2,1),(-1,-1),"CENTER"),
        ("ALIGN",(3,1),(3,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),4.5),
        ("BOTTOMPADDING",(0,0),(-1,-1),4.5),
    ]))
    story.append(rt)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Interpretation:</b> A high Threshold Events count indicates frequent threshold crossing. "
        "Interpret it together with event duration and the longest continuous excursion; it is not a count of failures.",
        note
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Report note:</b> This report is an analytical screening based on the uploaded Testo measurement data and the PPHG thresholds configured in the application. "
        "Operational records and applicable SOP requirements should be reviewed before corrective-action conclusions are made.",
        note
    ))

    doc.build(story, onFirstPage=_pdf_header_footer, onLaterPages=_pdf_header_footer)
    return buf.getvalue()




@st.cache_data(show_spinner=False, ttl=3600)
def generate_pdf_cached(result_json, median_minutes, analysis_start, analysis_end):
    """Cache generated PDF so repeated downloads do not rebuild the report."""
    result_df = pd.read_json(result_json, orient="split")
    median_interval = pd.Timedelta(minutes=float(median_minutes))
    period_data = pd.DataFrame({
        "Timestamp": pd.to_datetime([analysis_start, analysis_end])
    })
    return build_pdf_report(result_df, period_data, median_interval, raw=None)

uploaded = st.file_uploader("Upload Testo CSV", type=["csv"])

if uploaded:
    try:
        with st.spinner("Reading Testo CSV and analyzing equipment..."):
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
        if len(trend) > 2000:
            step = max(1, len(trend) // 2000)
            trend = trend.iloc[::step].copy()

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
        st.markdown("---")
        st.subheader("📄 PDF Report")
        st.caption("Generate the PDF only when needed to keep the dashboard responsive.")
        if st.button("📄 Generate PDF Report", type="primary", width="stretch", key="generate_pphg_pdf"):
            try:
                with st.spinner("Generating PDF report..."):
                    pdf_bytes = generate_pdf_cached(
                        result.to_json(orient="split", date_format="iso"),
                        float(median_interval.total_seconds() / 60.0),
                        str(pd.to_datetime(data["Timestamp"]).min()),
                        str(pd.to_datetime(data["Timestamp"]).max()),
                    )
                st.session_state["pphg_pdf_bytes"] = pdf_bytes
                st.success("PDF report siap di-download.")
            except Exception as pdf_error:
                st.session_state.pop("pphg_pdf_bytes", None)
                st.error(f"PDF report gagal dibuat: {pdf_error}")

        if "pphg_pdf_bytes" in st.session_state:
            st.download_button(
                label="⬇️ Download PPHG PDF Report",
                data=st.session_state["pphg_pdf_bytes"],
                file_name="PPHG_Temperature_Analysis_Report.pdf",
                mime="application/pdf",
                width="stretch",
                key="download_pphg_pdf",
            )
    except Exception as e:
        st.error(f"CSV tidak dapat diproses: {e}")
