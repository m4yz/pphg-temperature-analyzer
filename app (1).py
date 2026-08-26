import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="PPHG Temperature Analyzer", page_icon="🌡️", layout="wide")
st.title("🌡️ PPHG Temperature Analyzer")
st.caption("Upload Testo CSV → automatic PPHG temperature analysis")

RULES = {
    "Chiller": {"limit": 6.0, "delay": pd.Timedelta(hours=2)},
    "Freezer": {"limit": -15.0, "delay": pd.Timedelta(hours=4)},
}

def category(name):
    n = str(name).lower()
    if "freezer" in n: return "Freezer"
    if "chiller" in n: return "Chiller"
    return "Unknown"

def pick_col(df, names):
    m = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in m: return m[n.lower()]
    return None

def load_data(raw):
    tc = pick_col(raw, ["Time stamp","Timestamp","Time","Date Time","Datetime"])
    ec = pick_col(raw, ["Equipment","Measurement site","Device","Name"])
    vc = pick_col(raw, ["Temperature [°C]","Temperature","Temp","Value"])
    if not all([tc, ec, vc]):
        raise ValueError("CSV harus mempunyai kolom timestamp, equipment/device, dan temperature.")
    d = raw[[tc,ec,vc]].copy()
    d.columns = ["Timestamp","Equipment","Temperature"]
    d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="coerce")
    d["Temperature"] = pd.to_numeric(d["Temperature"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    d["Equipment"] = d["Equipment"].astype(str).str.strip()
    d = d.dropna(subset=["Timestamp","Equipment","Temperature"]).sort_values(["Equipment","Timestamp"])
    d["Category"] = d["Equipment"].map(category)
    return d

def analyze(g):
    g = g.sort_values("Timestamp").reset_index(drop=True)
    cat = g.Category.iloc[0]
    if cat not in RULES: return None
    limit, delay = RULES[cat]["limit"], RULES[cat]["delay"]
    above = g.Temperature >= limit
    # Episode IDs: every False->True starts a new continuous episode.
    starts = above.ne(above.shift(fill_value=False))
    eid = starts.cumsum()
    episodes = []
    for _, ep in g[above].groupby(eid[above]):
        start, end = ep.Timestamp.iloc[0], ep.Timestamp.iloc[-1]
        # Data is normally 5-minute Testo readings. Count the final observed
        # monitoring interval when it is followed by a reading.
        pos = g.index[g.Timestamp.eq(end)][0]
        duration = end - start
        if pos + 1 < len(g):
            nxt = g.Timestamp.iloc[pos+1]
            if nxt > end: duration = nxt - start
        episodes.append((duration,start,end,ep.Temperature.max()))
    if episodes:
        longest = max(episodes, key=lambda x:x[0])
        duration,start,end,peak = longest
        status = "ALARM" if duration >= delay else "WARNING"
    else:
        duration,start,end,peak = pd.Timedelta(0),pd.NaT,pd.NaT,g.Temperature.max()
        status = "NORMAL"
    return {
        "Equipment":g.Equipment.iloc[0], "Category":cat,
        "Min °C":g.Temperature.min(), "Average °C":g.Temperature.mean(),
        "Max °C":g.Temperature.max(),
        "Alarm Limit":f"≥{limit:g}°C / {delay.total_seconds()/3600:g}h",
        "Longest Continuous":duration, "Longest Start":start, "Longest End":end,
        "Peak During Excursion °C":peak, "Status":status
    }

def fmt(td):
    if pd.isna(td): return "—"
    mins = int(round(td.total_seconds()/60))
    d, r = divmod(mins,1440); h,m = divmod(r,60)
    return f"{d}d {h}h {m}m" if d else (f"{h}h {m}m" if h else f"{m}m")

up = st.file_uploader("Upload Testo CSV", type=["csv"])
if up:
    try:
        df = load_data(pd.read_csv(up))
        rows = [r for _,g in df.groupby("Equipment", sort=True) if (r:=analyze(g))]
        if not rows:
            st.warning("Tidak ada equipment Chiller/Freezer yang dikenali.")
            st.stop()
        out = pd.DataFrame(rows)
        order = {"ALARM":0,"WARNING":1,"NORMAL":2}
        out["_o"] = out.Status.map(order)
        out = out.sort_values(["_o","Longest Continuous"], ascending=[True,False]).drop(columns="_o")
        a,w,n = [(out.Status==s).sum() for s in ["ALARM","WARNING","NORMAL"]]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Equipment",len(out)); c2.metric("🔴 Alarm",a); c3.metric("🟠 Warning",w); c4.metric("🟢 Normal",n)
        show=out.copy()
        show["Longest Continuous"]=show["Longest Continuous"].map(fmt)
        for c in ["Min °C","Average °C","Max °C","Peak During Excursion °C"]: show[c]=show[c].round(1)
        st.subheader("PPHG Analysis")
        st.dataframe(show[["Equipment","Category","Min °C","Average °C","Max °C","Alarm Limit","Longest Continuous","Status"]],use_container_width=True,hide_index=True)
        st.subheader("Equipment Detail")
        sel=st.selectbox("Select equipment",out.Equipment.tolist())
        sd=df[df.Equipment==sel]
        sr=out[out.Equipment==sel].iloc[0]
        fig=px.line(sd,x="Timestamp",y="Temperature",title=f"{sel} — Temperature Trend")
        fig.add_hline(y=RULES[sr.Category]["limit"],line_dash="dash",annotation_text=f"PPHG limit {RULES[sr.Category]['limit']:g}°C")
        st.plotly_chart(fig,use_container_width=True)
        export=out.copy(); export["Longest Continuous"]=export["Longest Continuous"].map(fmt); export=export.drop(columns=["_o"],errors="ignore")
        st.download_button("Download Analysis CSV",export.to_csv(index=False).encode("utf-8"),"pphg_temperature_analysis.csv","text/csv")
    except Exception as e:
        st.error(f"CSV tidak dapat diproses: {e}")
else:
    st.info("Upload CSV Testo untuk mulai analisa.")
