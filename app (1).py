import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="PPHG Temperature Analyzer",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ PPHG Temperature Analyzer")
st.caption("Testo CSV → PPHG temperature analysis")

# PPHG SOP rules supplied by user
RULES = {
    "Chiller": {"limit": 6.0, "delay": pd.Timedelta(hours=2)},
    "Freezer": {"limit": -15.0, "delay": pd.Timedelta(hours=4)},
}

# Testo exports the 20 measurement points in this order.
# The location prefix is added here because Testo itself has duplicate
# measurement names such as "Upright Chiller 1".
TESTO_EQUIPMENT = [
    ("Eden Bar", "Chiller"),
    ("RL Kitchen – Showcase Chiller", "Chiller"),
    ("RL Kitchen – Upright Chiller 1", "Chiller"),
    ("RL Kitchen – Upright Chiller 2", "Chiller"),
    ("RL Kitchen – Upright Chiller 3", "Chiller"),
    ("RL Kitchen – Upright Freezer (2 Drawer)", "Freezer"),
    ("RL Kitchen – Undercounter Chiller", "Chiller"),
    ("RL Kitchen – Upright Chiller (2 Drawer)", "Chiller"),
    ("RL Kitchen – Freezer GEA 1", "Freezer"),
    ("RL Kitchen – Freezer GEA 3", "Freezer"),
    ("RL Kitchen – Freezer GEA 2", "Freezer"),
    ("Receiving – Freezer GEA 013", "Freezer"),
    ("Receiving – Freezer GEA 012", "Freezer"),
    ("Receiving – Freezer GEA 015", "Freezer"),
    ("Receiving – Showcase Chiller 014", "Chiller"),
    ("Kitchen 1 L90 – Upright Chiller 1 019", "Chiller"),
    ("Kitchen 1 L90 – Upright Chiller 2 020", "Chiller"),
    ("Kitchen 1 L90 – Upright Chiller 3 021", "Chiller"),
    ("Kitchen 1 L90 – Upright Freezer 1 022", "Freezer"),
    ("Canteen – Chiller", "Chiller"),
]

def read_testo_csv(uploaded_file):
    """
    Testo CSV format:
    - semicolon separated
    - first column = 'Asia/Jakarta GMT+7 (UTC +07:00)'
    - measurement columns = '<name>: Temperature (°C)'
    """
    raw = pd.read_csv(
        uploaded_file,
        sep=";",
        encoding="utf-8-sig",
        engine="python",
    )

    if raw.shape[1] < 2:
        raise ValueError("CSV Testo tidak terbaca sebagai semicolon-separated CSV.")

    time_col = raw.columns[0]
    timestamp = pd.to_datetime(raw[time_col], errors="coerce")

    if timestamp.notna().sum() == 0:
        raise ValueError("Kolom timestamp Testo tidak dapat dibaca.")

    # The first 20 measurement columns are the equipment in the report.
    measurement_cols = list(raw.columns[1:21])

    if len(measurement_cols) < 20:
        raise ValueError(
            f"CSV hanya memiliki {len(measurement_cols)} measurement columns; "
            "20 equipment yang diharapkan tidak lengkap."
        )

    frames = []

    for i, ((equipment, cat), col) in enumerate(
        zip(TESTO_EQUIPMENT, measurement_cols)
    ):
        temp = pd.to_numeric(
            raw[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

        part = pd.DataFrame({
            "Timestamp": timestamp,
            "Equipment": equipment,
            "Category": cat,
            "Temperature": temp,
        })

        part = part.dropna(subset=["Timestamp", "Temperature"])
        frames.append(part)

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["Equipment", "Timestamp"]).reset_index(drop=True)

    # Detect the actual sampling interval. Testo is normally 5 minutes.
    intervals = (
        data.groupby("Equipment")["Timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )
    typical_interval = intervals[intervals > 0].median()

    if pd.isna(typical_interval):
        typical_interval = 5.0

    return data, raw, typical_interval


def find_longest_continuous(data, category, typical_interval_minutes):
    """
    Find ONE continuous episode at/above the PPHG alarm threshold.

    Important:
    - Chiller: Temperature >= 6°C
    - Freezer: Temperature >= -15°C
    - Timer resets when temperature falls below the threshold.
    - Timer also resets if a data gap is larger than the normal Testo
      sampling interval. This avoids treating missing data as continuous.
    """
    limit = RULES[category]["limit"]
    delay = RULES[category]["delay"]

    g = data.sort_values("Timestamp").reset_index(drop=True).copy()
    g["AboveLimit"] = g["Temperature"] >= limit

    # Testo data is expected every 5 minutes. Allow a tiny tolerance for
    # timestamp rounding, but treat a real gap as a break in continuity.
    expected = pd.Timedelta(minutes=float(typical_interval_minutes))
    gap = g["Timestamp"].diff() > expected * 1.5

    # New episode whenever:
    # 1) temperature changes from below -> above, or
    # 2) there is a data gap.
    new_episode = (
        g["AboveLimit"]
        & (
            ~g["AboveLimit"].shift(1, fill_value=False)
            | gap.fillna(False)
        )
    )

    episode_id = new_episode.cumsum()
    g["EpisodeID"] = episode_id.where(g["AboveLimit"], -1)

    episodes = []

    for eid, ep in g[g["AboveLimit"]].groupby("EpisodeID"):
        start = ep["Timestamp"].iloc[0]
        end = ep["Timestamp"].iloc[-1]

        # Duration is the elapsed time between the first and last readings
        # that are continuously at/above the threshold.
        duration = end - start

        episodes.append({
            "Start": start,
            "End": end,
            "Duration": duration,
            "Peak": ep["Temperature"].max(),
        })

    if not episodes:
        return {
            "Longest Continuous": pd.Timedelta(0),
            "Longest Start": pd.NaT,
            "Longest End": pd.NaT,
            "Peak During Excursion °C": g["Temperature"].max(),
            "Status": "NORMAL",
        }

    longest = max(episodes, key=lambda x: x["Duration"])

    if longest["Duration"] >= delay:
        status = "ALARM"
    else:
        status = "WARNING"

    return {
        "Longest Continuous": longest["Duration"],
        "Longest Start": longest["Start"],
        "Longest End": longest["End"],
        "Peak During Excursion °C": longest["Peak"],
        "Status": status,
    }


def analyze(data, typical_interval_minutes):
    rows = []

    for equipment, g in data.groupby("Equipment", sort=False):
        category = g["Category"].iloc[0]
        rule = RULES[category]

        excursion = find_longest_continuous(
            g, category, typical_interval_minutes
        )

        exceeded_by = (
            max(excursion["Longest Continuous"] - rule["delay"], pd.Timedelta(0))
            if excursion["Status"] == "ALARM"
            else pd.Timedelta(0)
        )

        rows.append({
            "Equipment": equipment,
            "Category": category,
            "Min °C": g["Temperature"].min(),
            "Average °C": g["Temperature"].mean(),
            "Max °C": g["Temperature"].max(),
            "Alarm Limit": (
                f"≥{rule['limit']:g}°C / "
                f"{rule['delay'].total_seconds()/3600:g}h"
            ),
            "Exceeded By": exceeded_by,
            **excursion,
        })

    result = pd.DataFrame(rows)

    status_order = {"ALARM": 0, "WARNING": 1, "NORMAL": 2}
    result["_status_order"] = result["Status"].map(status_order)

    result = (
        result
        .sort_values(
            ["_status_order", "Longest Continuous"],
            ascending=[True, False],
        )
        .drop(columns="_status_order")
        .reset_index(drop=True)
    )

    return result


def format_duration(value):
    if pd.isna(value):
        return "—"

    minutes = int(round(value.total_seconds() / 60))
    days, remainder = divmod(minutes, 1440)
    hours, mins = divmod(remainder, 60)

    if days:
        return f"{days}d {hours}h {mins}m"
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"


uploaded = st.file_uploader(
    "Upload Testo CSV",
    type=["csv"],
    help="Gunakan CSV hasil export langsung dari Testo.",
)

if uploaded:
    try:
        data, raw, typical_interval = read_testo_csv(uploaded)

        result = analyze(data, typical_interval)

        alarm_count = int((result["Status"] == "ALARM").sum())
        warning_count = int((result["Status"] == "WARNING").sum())
        normal_count = int((result["Status"] == "NORMAL").sum())

        st.success(
            f"Testo CSV berhasil dibaca: {len(result)} equipment dianalisis. "
            f"Sampling interval terdeteksi ≈ {typical_interval:g} menit."
        )

        # Testo file supplied has 4 additional measurement columns after the
        # 20 equipment in this report. They are intentionally not included in
        # the PPHG table until their equipment/category mapping is defined.
        extra_count = max(0, raw.shape[1] - 21)
        if extra_count:
            st.info(
                f"{extra_count} kolom measurement tambahan di file Testo "
                "tidak dimasukkan ke tabel PPHG karena belum ada mapping "
                "equipment/category pada report ini."
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Equipment", len(result))
        c2.metric("🔴 Alarm", alarm_count)
        c3.metric("🟠 Warning", warning_count)
        c4.metric("🟢 Normal", normal_count)

        st.divider()
        st.subheader("PPHG Analysis")

        display = result.copy()

        for col in ["Min °C", "Average °C", "Max °C"]:
            display[col] = display[col].round(1)

        display["Longest Continuous"] = display[
            "Longest Continuous"
        ].apply(format_duration)

        display["Longest Start"] = pd.to_datetime(
            display["Longest Start"], errors="coerce"
        ).dt.strftime("%d-%m-%Y %H:%M").fillna("—")

        display["Longest End"] = pd.to_datetime(
            display["Longest End"], errors="coerce"
        ).dt.strftime("%d-%m-%Y %H:%M").fillna("—")

        display["Exceeded By"] = display["Exceeded By"].apply(
            lambda x: format_duration(x) if x > pd.Timedelta(0) else "—"
        )

        table_cols = [
            "Equipment",
            "Category",
            "Min °C",
            "Average °C",
            "Max °C",
            "Alarm Limit",
            "Longest Start",
            "Longest End",
            "Longest Continuous",
            "Exceeded By",
            "Status",
        ]

        def color_status(row):
            styles = [""] * len(row)
            status_idx = table_cols.index("Status")
            if row["Status"] == "ALARM":
                styles[status_idx] = "background-color: #ffd6d6; color: #b00020; font-weight: 700"
            elif row["Status"] == "WARNING":
                styles[status_idx] = "background-color: #fff0cc; color: #9a5b00; font-weight: 700"
            elif row["Status"] == "NORMAL":
                styles[status_idx] = "background-color: #d9f5e5; color: #087443; font-weight: 700"
            return styles

        styled_display = (
            display[table_cols]
            .style
            .apply(color_status, axis=1)
        )

        st.dataframe(
            styled_display,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Equipment Detail")

        selected = st.selectbox(
            "Select equipment",
            result["Equipment"].tolist(),
        )

        selected_data = data[data["Equipment"] == selected].copy()
        selected_result = result[result["Equipment"] == selected].iloc[0]
        category = selected_result["Category"]
        limit = RULES[category]["limit"]

        fig = px.line(
            selected_data,
            x="Timestamp",
            y="Temperature",
            title=f"{selected} — Temperature Trend",
        )

        fig.add_hline(
            y=limit,
            line_dash="dash",
            annotation_text=f"PPHG limit {limit:g}°C",
        )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write(
            "**Longest Continuous Time ≥ Alarm Limit:** "
            f"{format_duration(selected_result['Longest Continuous'])}"
        )

        export = result.copy()
        export["Longest Continuous"] = export[
            "Longest Continuous"
        ].apply(format_duration)

        d1, d2, d3 = st.columns(3)

        d1.metric(
            "Longest Continuous",
            format_duration(selected_result["Longest Continuous"]),
        )

        d2.metric(
            "Excursion Start",
            (
                selected_result["Longest Start"].strftime("%d-%m-%Y %H:%M")
                if pd.notna(selected_result["Longest Start"])
                else "—"
            ),
        )

        d3.metric(
            "Excursion End",
            (
                selected_result["Longest End"].strftime("%d-%m-%Y %H:%M")
                if pd.notna(selected_result["Longest End"])
                else "—"
            ),
        )

        st.download_button(
            "Download Analysis CSV",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name="pphg_temperature_analysis.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error(f"CSV tidak dapat diproses: {exc}")
else:
    st.info("Upload CSV Testo untuk mulai analisa.")
