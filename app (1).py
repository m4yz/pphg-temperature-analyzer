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
    interval = df["Timestamp"].sort_values().diff().dropna()
    median_interval = interval.median() if not interval.empty else pd.Timedelta(minutes=5)

    for equipment, g in df.groupby("Equipment", sort=False):
        category = g["Category"].iloc[0]
        duration, start, end, peak, status = longest_continuous(
            g, category, median_interval
        )

        if category in RULES:
            rule = RULES[category]
            exceeded = max(duration - rule["delay"], pd.Timedelta(0))
            limit_text = (
                f"≥{rule['limit']:g}°C / "
                f"{rule['delay'].total_seconds()/3600:g}h"
            )
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
            "Longest Start": start,
            "Longest End": end,
            "Longest Continuous": duration,
            "Exceeded By": exceeded,
            "Status": status,
            "Peak During Excursion °C": peak,
        })

    out = pd.DataFrame(rows)
    order = {"ALARM": 0, "WARNING": 1, "NORMAL": 2, "N/A": 3}
    out["_order"] = out["Status"].map(order).fillna(9)
    out = out.sort_values(
        ["_order", "Longest Continuous"],
        ascending=[True, False]
    ).drop(columns="_order")
    return out, median_interval


def build_pdf_report(result, data, median_interval):
    """Build a management-friendly PPHG temperature analysis PDF."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=10*mm,
        leftMargin=10*mm,
        topMargin=10*mm,
        bottomMargin=10*mm,
        title="PPHG Temperature Analysis Report",
        author="PPHG Temperature Analyzer",
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=20,
        leading=24, textColor=colors.HexColor("#17324D"),
        alignment=TA_LEFT, spaceAfter=5*mm
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"], fontSize=13,
        leading=16, textColor=colors.HexColor("#17324D"),
        spaceBefore=4*mm, spaceAfter=3*mm
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=8.5,
        leading=12, spaceAfter=2*mm
    )
    small = ParagraphStyle(
        "Small", parent=styles["BodyText"], fontSize=7,
        leading=9
    )
    note = ParagraphStyle(
        "Note", parent=styles["BodyText"], fontSize=8,
        leading=11, textColor=colors.HexColor("#555555")
    )

    story = []
    start = data["Timestamp"].min()
    end = data["Timestamp"].max()
    total = len(result)
    alarms = int((result["Status"] == "ALARM").sum())
    warnings = int((result["Status"] == "WARNING").sum())
    normal = int((result["Status"] == "NORMAL").sum())
    other = int((result["Category"] == "Other").sum())

    story.append(Paragraph("PPHG Temperature Analysis Report", title))
    story.append(Paragraph(
        f"<b>Measurement period:</b> {start:%d-%m-%Y %H:%M} to {end:%d-%m-%Y %H:%M}"
        f"&nbsp;&nbsp;&nbsp; <b>Sampling interval:</b> "
        f"≈ {round(median_interval.total_seconds()/60):g} minutes",
        body
    ))

    # Executive summary
    story.append(Paragraph("1. Executive Summary", h1))
    summary_data = [
        ["Equipment", "Alarm", "Warning", "Normal", "Other / N/A"],
        [str(total), str(alarms), str(warnings), str(normal), str(other)],
    ]
    t = Table(summary_data, colWidths=[35*mm]*5)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17324D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#B8C2CC")),
        ("BACKGROUND", (0,1), (-1,1), colors.HexColor("#F4F7F9")),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 3*mm))

    if alarms:
        alarm_df = result[result["Status"] == "ALARM"].copy()
        alarm_df["excess_min"] = alarm_df["Exceeded By"].dt.total_seconds()/60
        alarm_df = alarm_df.sort_values(
            ["excess_min", "Longest Continuous"], ascending=[False, False]
        )
        top = alarm_df.head(5)
        bullets = [
            f"<b>{r['Equipment']}</b>: continuous excursion "
            f"{format_duration(r['Longest Continuous'])}, exceeded PPHG delay by "
            f"<b>{format_duration(r['Exceeded By'])}</b>, peak "
            f"{r['Peak During Excursion °C']:.1f}°C."
            for _, r in top.iterrows()
        ]
        story.append(Paragraph(
            "<b>Key finding:</b> "
            f"{alarms} equipment meet the PPHG alarm-duration criterion. "
            "The five most significant excursions are:",
            body
        ))
        for b in bullets:
            story.append(Paragraph("• " + b, body))
    elif warnings:
        story.append(Paragraph(
            f"<b>Key finding:</b> No equipment reached the PPHG alarm-duration "
            f"criterion, but {warnings} equipment remain in warning status.",
            body
        ))
    else:
        story.append(Paragraph(
            "No PPHG Chiller/Freezer alarm-duration excursions were identified.",
            body
        ))

    if other:
        story.append(Paragraph(
            f"<b>Data limitation:</b> {other} measurement(s) are classified as "
            "<b>Other</b> and are not assigned a Chiller/Freezer PPHG delay rule. "
            "These should be mapped after Testo naming is made unique.",
            note
        ))

    # Methodology
    story.append(Paragraph("2. PPHG Assessment Criteria", h1))
    criteria = [
        ["Category", "Temperature excursion", "Alarm delay", "Interpretation"],
        ["Chiller", "≥ 6°C", "2 hours", "ALARM when continuous excursion ≥ 2h"],
        ["Freezer", "≥ -15°C", "4 hours", "ALARM when continuous excursion ≥ 4h"],
    ]
    ct = Table(criteria, colWidths=[35*mm, 45*mm, 35*mm, 105*mm])
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17324D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#B8C2CC")),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(ct)
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "Longest Continuous is calculated from consecutive samples above the "
        "applicable PPHG temperature threshold. A sampling gap is not treated as "
        "continuous excursion. Exceeded By is the portion of the longest "
        "continuous excursion beyond the applicable PPHG delay.",
        note
    ))

    # Full analysis table
    story.append(PageBreak())
    story.append(Paragraph("3. Equipment Analysis", h1))

    headers = [
        "Equipment", "Cat.", "Min °C", "Avg °C", "Max °C",
        "Limit", "Start", "End", "Continuous", "Exceeded", "Status"
    ]
    table_data = [headers]
    for _, r in result.iterrows():
        start_txt = r["Longest Start"].strftime("%d-%m %H:%M") if pd.notna(r["Longest Start"]) else "—"
        end_txt = r["Longest End"].strftime("%d-%m %H:%M") if pd.notna(r["Longest End"]) else "—"
        table_data.append([
            Paragraph(str(r["Equipment"]), small),
            str(r["Category"]),
            f"{r['Min °C']:.1f}",
            f"{r['Average °C']:.1f}",
            f"{r['Max °C']:.1f}",
            str(r["Alarm Limit"]),
            start_txt,
            end_txt,
            format_duration(r["Longest Continuous"]),
            format_duration(r["Exceeded By"]) if r["Exceeded By"] > pd.Timedelta(0) else "—",
            str(r["Status"]),
        ])

    col_widths = [48*mm, 17*mm, 15*mm, 15*mm, 15*mm, 27*mm,
                  25*mm, 25*mm, 27*mm, 24*mm, 18*mm]
    at = Table(table_data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17324D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 6.7),
        ("FONTSIZE", (0,1), (-1,-1), 6.5),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#C5CCD3")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (2,1), (9,-1), "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]
    for i, (_, r) in enumerate(result.iterrows(), start=1):
        if r["Status"] == "ALARM":
            style_cmds.append(("BACKGROUND", (-1,i), (-1,i), colors.HexColor("#FFD6D6")))
            style_cmds.append(("TEXTCOLOR", (-1,i), (-1,i), colors.HexColor("#B00020")))
            style_cmds.append(("FONTNAME", (-1,i), (-1,i), "Helvetica-Bold"))
        elif r["Status"] == "WARNING":
            style_cmds.append(("BACKGROUND", (-1,i), (-1,i), colors.HexColor("#FFF0CC")))
        elif r["Status"] == "NORMAL":
            style_cmds.append(("BACKGROUND", (-1,i), (-1,i), colors.HexColor("#D9F5E5")))
    at.setStyle(TableStyle(style_cmds))
    story.append(at)

    # Analysis / recommendations
    story.append(PageBreak())
    story.append(Paragraph("4. Management Analysis & Recommended Actions", h1))

    if alarms:
        story.append(Paragraph(
            "The equipment listed as ALARM should be prioritized for operational "
            "review because the longest continuous excursion exceeded the PPHG "
            "delay requirement. Priority should generally be given to equipment "
            "with the largest 'Exceeded By' duration and the highest excursion peak.",
            body
        ))
        for _, r in alarm_df.head(10).iterrows():
            story.append(Paragraph(
                f"• <b>{r['Equipment']}</b> — {r['Category']}; "
                f"continuous {format_duration(r['Longest Continuous'])}; "
                f"exceeded by {format_duration(r['Exceeded By'])}; "
                f"peak {r['Peak During Excursion °C']:.1f}°C. "
                f"Review door opening, loading, ambient exposure, condenser/coil "
                f"condition, and equipment performance as applicable.",
                body
            ))

    if warnings:
        story.append(Paragraph(
            "WARNING equipment did not exceed the PPHG delay duration, but the "
            "temperature excursion was still detected. These units should be "
            "reviewed for recurring short excursions, especially where the "
            "Longest Continuous duration is approaching the alarm delay.",
            body
        ))

    if other:
        story.append(Paragraph(
            "Measurement points currently classified as Other are excluded from "
            "PPHG Chiller/Freezer alarm classification. After unique Testo naming "
            "is implemented, their category should be mapped before using the "
            "report for formal compliance assessment.",
            body
        ))

    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        "<b>Report note:</b> This report is an analytical screening based on the "
        "uploaded Testo measurement data and the PPHG SOP thresholds configured "
        "in the application. It should be reviewed together with operational "
        "records and any applicable SOP requirements before formal corrective "
        "action or compliance conclusions.",
        note
    ))

    doc.build(story)
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

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Equipment", equipment_count)
        c2.metric("🔴 Alarm", alarms)
        c3.metric("🟠 Warning", warnings)
        c4.metric("🟢 Normal", normal)

        st.divider()
        st.subheader("PPHG Analysis")

        show = result.copy()
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
            "Alarm Limit", "Longest Start", "Longest End",
            "Longest Continuous", "Exceeded By", "Status"
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
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.subheader("Equipment Detail")

        selected = st.selectbox(
            "Select equipment",
            result["Equipment"].tolist()
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

        st.plotly_chart(fig, use_container_width=True)

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

        export = result.copy()
        for col in ["Longest Continuous", "Exceeded By"]:
            export[col] = export[col].map(
                lambda x: format_duration(x) if pd.notna(x) else "—"
            )
        export["Longest Start"] = pd.to_datetime(
            export["Longest Start"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M")
        export["Longest End"] = pd.to_datetime(
            export["Longest End"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M")

        st.download_button(
            "Download Analysis CSV",
            export.to_csv(index=False).encode("utf-8-sig"),
            "pphg_temperature_analysis.csv",
            "text/csv",
        )

        pdf_bytes = build_pdf_report(result, data, median_interval)
        st.download_button(
            "📄 Download PPHG PDF Report",
            pdf_bytes,
            "PPHG_Temperature_Analysis_Report.pdf",
            "application/pdf",
        )

    except Exception as e:
        st.error(f"CSV tidak dapat diproses: {e}")
