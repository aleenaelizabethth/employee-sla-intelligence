import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Employee SLA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT   = "#00E5BE"
ACCENT2  = "#FF6B6B"
BG_DARK  = "#0D1117"
BG_CARD  = "#161B22"
BG_CARD2 = "#1C2230"
TEXT     = "#E6EDF3"
MUTED    = "#8B949E"

PLT_STYLE = {
    "figure.facecolor":  BG_CARD,
    "axes.facecolor":    BG_CARD2,
    "axes.edgecolor":    "#30363D",
    "axes.labelcolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        "#21262D",
    "grid.linewidth":    0.6,
    "axes.grid":         True,
    "font.family":       "monospace",
}
plt.rcParams.update(PLT_STYLE)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  html, body, [class*="css"] {{
      background-color: {BG_DARK};
      color: {TEXT};
      font-family: 'Syne', sans-serif;
  }}
  .block-container {{ padding: 2rem 2.5rem 3rem; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
      background: {BG_CARD};
      border-right: 1px solid #21262D;
  }}
  [data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

  /* Headers */
  h1 {{ font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.4rem;
        background: linear-gradient(90deg, {ACCENT}, #6EE7F7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  h2, h3 {{ font-family: 'Syne', sans-serif; font-weight: 600; color: {TEXT}; }}

  /* Metric cards */
  .metric-card {{
      background: {BG_CARD};
      border: 1px solid #21262D;
      border-left: 4px solid {ACCENT};
      border-radius: 10px;
      padding: 1.2rem 1.4rem;
      margin-bottom: 0.6rem;
  }}
  .metric-label {{ font-size: 0.72rem; letter-spacing: 0.15em;
                   text-transform: uppercase; color: {MUTED}; }}
  .metric-value {{ font-size: 2rem; font-weight: 800;
                   font-family: 'Space Mono', monospace; color: {ACCENT}; }}
  .metric-sub   {{ font-size: 0.78rem; color: {MUTED}; margin-top: 2px; }}

  /* Section divider */
  .section-header {{
      display: flex; align-items: center; gap: 12px;
      margin: 2.2rem 0 1rem;
  }}
  .section-header .icon {{ font-size: 1.5rem; }}
  .section-header .title {{
      font-size: 1.15rem; font-weight: 700;
      letter-spacing: 0.05em; color: {TEXT};
  }}
  .section-line {{
      flex: 1; height: 1px;
      background: linear-gradient(90deg, #30363D, transparent);
  }}

  /* Status badges */
  .badge-met     {{ background:#0D4429; color:#3FB950; border-radius:5px;
                    padding:2px 10px; font-size:0.8rem; font-weight:700; }}
  .badge-breach  {{ background:#4D1010; color:{ACCENT2}; border-radius:5px;
                    padding:2px 10px; font-size:0.8rem; font-weight:700; }}

  /* Dataframe */
  [data-testid="stDataFrame"] {{ border-radius: 8px; }}

  /* Buttons */
  .stButton > button {{
      background: {ACCENT}; color: #000; border: none;
      font-weight: 700; border-radius: 8px; padding: 0.55rem 1.8rem;
      font-family: 'Syne', sans-serif; letter-spacing: 0.05em;
      transition: opacity .2s;
  }}
  .stButton > button:hover {{ opacity: 0.85; }}

  /* Tabs */
  [data-testid="stTabs"] button {{
      color: {MUTED} !important; font-family: 'Syne', sans-serif !important;
      font-weight: 600 !important;
  }}
  [data-testid="stTabs"] [aria-selected="true"] {{
      color: {ACCENT} !important;
      border-bottom: 2px solid {ACCENT} !important;
  }}

  /* Info/success boxes */
  .stAlert {{ border-radius: 8px; }}

  /* Prediction result */
  .pred-box {{
      background: {BG_CARD}; border-radius: 12px;
      border: 2px solid {ACCENT}; padding: 1.5rem 2rem;
      text-align: center; margin-top: 1rem;
  }}
  .pred-title {{ font-size: 0.85rem; letter-spacing: 0.2em;
                 text-transform: uppercase; color: {MUTED}; }}
  .pred-result {{ font-size: 2.6rem; font-weight: 800;
                  font-family: 'Space Mono', monospace; }}
  .pred-met     {{ color: #3FB950; }}
  .pred-breach  {{ color: {ACCENT2}; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def section(icon, title):
    st.markdown(f"""
    <div class="section-header">
      <span class="icon">{icon}</span>
      <span class="title">{title}</span>
      <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

def metric_card(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

CMAP_MAIN = [ACCENT, "#6EE7F7", "#A78BFA", ACCENT2, "#FBBF24", "#34D399"]


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h2 style='color:{ACCENT};margin-bottom:0'>⚡ SLA Analytics</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};font-size:0.8rem;margin-top:4px'>Employee Performance Intelligence</p>", unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader("Upload employee_sla_status.csv", type=["csv"])
    st.divider()
    st.markdown(f"<p style='color:{MUTED};font-size:0.75rem'>Built with KNN · scikit-learn<br>Streamlit Dashboard v1.0</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.replace("", pd.NA, inplace=True)
    return df

if uploaded is None:
    st.markdown("<h1>Employee SLA Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{BG_CARD};border:1px dashed #30363D;border-radius:12px;
                padding:3rem;text-align:center;margin-top:2rem;">
      <div style="font-size:3.5rem">📂</div>
      <h3 style="color:{TEXT}">No data loaded yet</h3>
      <p style="color:{MUTED}">Upload <code>employee_sla_status.csv</code> from the sidebar to begin.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

df_raw = load_data(uploaded)


# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.markdown("<h1>Employee SLA Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{MUTED};margin-top:-1rem;margin-bottom:1.5rem'>Exploratory Analysis · Feature Engineering · KNN Classifier</p>", unsafe_allow_html=True)

# KPI row
total      = len(df_raw)
met_pct    = round((df_raw["SLA_Status"] == "Met").sum() / total * 100, 1) if "SLA_Status" in df_raw.columns else "—"
breach_pct = round(100 - met_pct, 1) if isinstance(met_pct, float) else "—"
teams      = df_raw["Team"].nunique() if "Team" in df_raw.columns else "—"
shifts     = df_raw["Shift"].nunique() if "Shift" in df_raw.columns else "—"
missing    = df_raw.isna().sum().sum()

k1, k2, k3, k4, k5 = st.columns(5)
with k1: metric_card("Total Records",   f"{total:,}",       "employees in dataset")
with k2: metric_card("SLA Met",         f"{met_pct}%",      "on-time performance")
with k3: metric_card("SLA Breached",    f"{breach_pct}%",   "requires attention")
with k4: metric_card("Teams",           str(teams),         "distinct departments")
with k5: metric_card("Missing Values",  str(missing),       "before imputation")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🗂️  Data Overview", "📊  Visualizations", "🤖  ML Model", "🔮  Predict"])

# ══════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns([2, 1])

    with c1:
        section("📋", "Raw Dataset")
        st.dataframe(df_raw, use_container_width=True, height=320)

    with c2:
        section("🧬", "Column Types")
        dtype_df = pd.DataFrame({
            "Column": df_raw.dtypes.index,
            "Type":   df_raw.dtypes.astype(str).values
        })
        st.dataframe(dtype_df, use_container_width=True, height=320, hide_index=True)

    c3, c4 = st.columns(2)
    with c3:
        section("🔍", "First 5 Rows")
        st.dataframe(df_raw.head(), use_container_width=True, hide_index=True)
    with c4:
        section("🔍", "Last 5 Rows")
        st.dataframe(df_raw.tail(), use_container_width=True, hide_index=True)

    section("🕳️", "Missing Values")
    miss = df_raw.isna().sum().reset_index()
    miss.columns = ["Column", "Missing Count"]
    miss["Missing %"] = (miss["Missing Count"] / len(df_raw) * 100).round(2)
    miss = miss[miss["Missing Count"] > 0]
    if miss.empty:
        st.success("✅ No missing values found in the dataset.")
    else:
        st.dataframe(miss, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 2 — VISUALIZATIONS
# ══════════════════════════════════════════════
with tab2:
    # Row 1: Team & Shift distributions
    section("📊", "Distribution Analysis")
    v1, v2 = st.columns(2)

    with v1:
        team_counts = df_raw["Team"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        bars = ax.bar(team_counts.index, team_counts.values,
                      color=CMAP_MAIN[:len(team_counts)], edgecolor="none", width=0.6)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(int(bar.get_height())), ha="center", va="bottom",
                    fontsize=9, color=TEXT, fontfamily="monospace")
        ax.set_title("Employees per Team", fontsize=12, fontweight="bold", pad=10, color=TEXT)
        ax.set_xlabel("Team", fontsize=9, labelpad=6)
        ax.set_ylabel("Count", fontsize=9, labelpad=6)
        ax.spines[:].set_visible(False)
        plt.xticks(rotation=20, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with v2:
        shift_counts = df_raw["Shift"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        wedges, texts, autotexts = ax.pie(
            shift_counts.values,
            labels=shift_counts.index,
            autopct="%1.1f%%",
            colors=[ACCENT, "#A78BFA", ACCENT2, "#FBBF24"][:len(shift_counts)],
            startangle=140,
            wedgeprops={"edgecolor": BG_DARK, "linewidth": 2},
            pctdistance=0.78,
        )
        for t in texts:   t.set_color(TEXT);    t.set_fontsize(9)
        for a in autotexts: a.set_color(BG_DARK); a.set_fontweight("bold"); a.set_fontsize(8)
        ax.set_title("Shift Distribution", fontsize=12, fontweight="bold", pad=10, color=TEXT)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # Row 2: SLA Status & SLA by Shift
    v3, v4 = st.columns(2)

    with v3:
        sla_counts = df_raw["SLA_Status"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        colors = [ACCENT if s == "Met" else ACCENT2 for s in sla_counts.index]
        bars = ax.bar(sla_counts.index, sla_counts.values,
                      color=colors, edgecolor="none", width=0.45)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(int(bar.get_height())), ha="center", va="bottom",
                    fontsize=10, color=TEXT, fontfamily="monospace", fontweight="bold")
        ax.set_title("SLA Status Distribution", fontsize=12, fontweight="bold", pad=10, color=TEXT)
        ax.set_xlabel("Status", fontsize=9); ax.set_ylabel("Count", fontsize=9)
        ax.spines[:].set_visible(False)
        met_patch    = mpatches.Patch(color=ACCENT,  label="Met")
        breach_patch = mpatches.Patch(color=ACCENT2, label="Breached")
        ax.legend(handles=[met_patch, breach_patch], fontsize=8,
                  facecolor=BG_CARD, edgecolor="#30363D", labelcolor=TEXT)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with v4:
        shift_sla = pd.crosstab(df_raw["Shift"], df_raw["SLA_Status"])
        fig, ax = plt.subplots(figsize=(6, 3.8))
        x = np.arange(len(shift_sla.index))
        w = 0.35
        cols = shift_sla.columns.tolist()
        bar_colors = {"Met": ACCENT, "Breached": ACCENT2}
        for i, col in enumerate(cols):
            ax.bar(x + i*w, shift_sla[col], width=w, label=col,
                   color=bar_colors.get(col, CMAP_MAIN[i]), edgecolor="none")
        ax.set_xticks(x + w/2); ax.set_xticklabels(shift_sla.index, fontsize=8, rotation=15, ha="right")
        ax.set_title("SLA Status by Shift", fontsize=12, fontweight="bold", pad=10, color=TEXT)
        ax.set_ylabel("Count", fontsize=9)
        ax.spines[:].set_visible(False)
        ax.legend(fontsize=8, facecolor=BG_CARD, edgecolor="#30363D", labelcolor=TEXT)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # Row 3: Correlation Heatmap & Boxplot
    section("🔬", "Statistical Insights")
    v5, v6 = st.columns(2)

    with v5:
        num_df = df_raw.select_dtypes(include=["int64","float64","number"])
        if not num_df.empty:
            cor = num_df.corr()
            fig, ax = plt.subplots(figsize=(6, 5))
            mask = np.triu(np.ones_like(cor, dtype=bool))
            sns.heatmap(cor, mask=mask, annot=True, cmap="viridis", center=0,
                        fmt=".2f", linewidths=0.5, ax=ax,
                        annot_kws={"size": 8, "color": TEXT},
                        cbar_kws={"shrink": 0.75})
            ax.set_title("Correlation Heatmap", fontsize=12, fontweight="bold", pad=10, color=TEXT)
            plt.xticks(fontsize=7, rotation=40, ha="right")
            plt.yticks(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for correlation.")

    with v6:
        if not num_df.empty:
            fig, ax = plt.subplots(figsize=(6, 5))
            bp = ax.boxplot(
                [num_df[c].dropna() for c in num_df.columns],
                labels=num_df.columns,
                patch_artist=True,
                medianprops={"color": ACCENT, "linewidth": 2},
                whiskerprops={"color": MUTED},
                capprops={"color": MUTED},
                flierprops={"marker":"o","markerfacecolor":ACCENT2,"markersize":4,"alpha":0.6},
            )
            colors = CMAP_MAIN[:len(num_df.columns)]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color); patch.set_alpha(0.4); patch.set_edgecolor(color)
            ax.set_title("Outlier Detection — Numeric Features", fontsize=12, fontweight="bold", pad=10, color=TEXT)
            ax.set_ylabel("Value", fontsize=9)
            plt.xticks(rotation=40, ha="right", fontsize=7)
            ax.spines[:].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — ML MODEL
# ══════════════════════════════════════════════
with tab3:
    section("⚙️", "Feature Engineering & Model Training")

    @st.cache_resource
    def build_model(df):
        df2 = df.copy()
        df2.replace("", pd.NA, inplace=True)

        df_enc = pd.get_dummies(df2[["Team","Shift"]], drop_first=True, dtype=int)
        df2["SLA_Status"] = df2["SLA_Status"].map({"Breached": 0, "Met": 1})

        dfe = pd.concat([df2, df_enc], axis=1)
        dfe.drop(["Employee_ID","Team","Shift"], axis=1, inplace=True, errors="ignore")

        for col in ["Avg_TAT_Hours", "Error_Rate", "Quality_Score"]:
            if col in dfe.columns:
                if dfe[col].dtype == object:
                    dfe[col] = pd.to_numeric(dfe[col], errors="coerce")
                if col == "Avg_TAT_Hours":
                    dfe[col].fillna(dfe[col].mean(), inplace=True)
                else:
                    dfe[col].fillna(dfe[col].median(), inplace=True)

        dfe.dropna(subset=["SLA_Status"], inplace=True)
        dfe = dfe.dropna()

        x = dfe.drop("SLA_Status", axis=1)
        y = dfe["SLA_Status"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        scaler = MinMaxScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s  = scaler.transform(x_test)

        # Base KNN
        base_model = KNeighborsClassifier(n_neighbors=5)
        base_model.fit(x_train_s, y_train)
        base_acc = accuracy_score(y_test, base_model.predict(x_test_s))

        # Grid Search
        grid = GridSearchCV(
            KNeighborsClassifier(),
            {"n_neighbors":[3,5,7,9,11], "weights":["uniform","distance"]},
            cv=5, scoring="accuracy", n_jobs=-1
        )
        grid.fit(x_train_s, y_train)

        best = grid.best_estimator_
        y_pred = best.predict(x_test_s)
        best_acc = accuracy_score(y_test, y_pred)
        report   = classification_report(y_test, y_pred, output_dict=True, target_names=["Breached","Met"])
        cm       = confusion_matrix(y_test, y_pred)

        return {
            "base_acc":    base_acc,
            "best_acc":    best_acc,
            "best_params": grid.best_params_,
            "report":      report,
            "cm":          cm,
            "scaler":      scaler,
            "model":       best,
            "features":    list(x.columns),
            "x_train":     x_train,
        }

    with st.spinner("Training model — please wait..."):
        results = build_model(df_raw)

    # KPI row
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("Base KNN Accuracy",  f"{results['base_acc']*100:.1f}%",  "n_neighbors=5")
    with m2: metric_card("Tuned KNN Accuracy", f"{results['best_acc']*100:.1f}%",  "after GridSearchCV")
    with m3: metric_card("Best n_neighbors",    str(results["best_params"]["n_neighbors"]), "optimal K value")
    with m4: metric_card("Best Weights",        results["best_params"]["weights"].title(), "distance or uniform")

    # Confusion Matrix + Classification Report
    section("📈", "Evaluation Results")
    e1, e2 = st.columns(2)

    with e1:
        st.markdown(f"<h3 style='font-size:0.95rem;color:{MUTED};letter-spacing:.1em'>CONFUSION MATRIX</h3>", unsafe_allow_html=True)
        cm = results["cm"]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                    xticklabels=["Breached","Met"], yticklabels=["Breached","Met"],
                    linewidths=1, linecolor=BG_DARK, ax=ax,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold", pad=10, color=TEXT)
        ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("Actual", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with e2:
        st.markdown(f"<h3 style='font-size:0.95rem;color:{MUTED};letter-spacing:.1em'>CLASSIFICATION REPORT</h3>", unsafe_allow_html=True)
        rep = results["report"]
        rows = []
        for label in ["Breached","Met","macro avg","weighted avg"]:
            if label in rep:
                r = rep[label]
                rows.append({
                    "Class":     label,
                    "Precision": round(r["precision"], 3),
                    "Recall":    round(r["recall"], 3),
                    "F1-Score":  round(r["f1-score"], 3),
                    "Support":   int(r["support"]),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=200)

        # Accuracy bar
        fig2, ax2 = plt.subplots(figsize=(5, 1.8))
        acc = results["best_acc"]
        ax2.barh(["Accuracy"], [acc], color=ACCENT, edgecolor="none", height=0.45)
        ax2.barh(["Accuracy"], [1 - acc], left=[acc], color="#21262D", edgecolor="none", height=0.45)
        ax2.text(acc/2, 0, f"{acc*100:.1f}%", va="center", ha="center",
                 fontsize=13, fontweight="bold", color=BG_DARK, fontfamily="monospace")
        ax2.set_xlim(0, 1); ax2.set_title("Model Accuracy", fontsize=10, color=TEXT, pad=8)
        ax2.spines[:].set_visible(False); ax2.set_yticks([]); ax2.set_xticks([])
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    # Feature importance (via permutation proxy: variance of training features)
    section("🧩", "Feature Columns Used")
    feat_df = pd.DataFrame({"Feature": results["features"], "Index": range(len(results["features"]))})
    st.dataframe(feat_df, use_container_width=True, hide_index=True, height=180)


# ══════════════════════════════════════════════
# TAB 4 — PREDICT
# ══════════════════════════════════════════════
with tab4:
    section("🔮", "Predict SLA Status for a New Employee")

    with st.spinner("Loading model..."):
        results = build_model(df_raw)

    st.markdown(f"<p style='color:{MUTED}'>Fill in the employee details below. The trained KNN model will predict whether their SLA will be <b style='color:{ACCENT}'>Met</b> or <b style='color:{ACCENT2}'>Breached</b>.</p>", unsafe_allow_html=True)

    num_cols = df_raw.select_dtypes(include=["int64","float64","number"]).columns.tolist()
    if "SLA_Status" in num_cols: num_cols.remove("SLA_Status")

    col_a, col_b = st.columns(2)
    user_input = {}

    # Numeric sliders
    half = len(num_cols) // 2 + len(num_cols) % 2
    for i, col in enumerate(num_cols):
        mn  = float(df_raw[col].min()) if pd.notna(df_raw[col].min()) else 0.0
        mx  = float(df_raw[col].max()) if pd.notna(df_raw[col].max()) else 100.0
        med = float(df_raw[col].median()) if pd.notna(df_raw[col].median()) else (mn+mx)/2
        target_col = col_a if i < half else col_b
        with target_col:
            user_input[col] = st.slider(col.replace("_"," "), mn, mx, med, step=(mx-mn)/200 if mx != mn else 1.0)

    # Categorical dropdowns
    cat_cols = ["Team","Shift"]
    for cat in cat_cols:
        if cat in df_raw.columns:
            options = sorted(df_raw[cat].dropna().unique().tolist())
            with col_a if cat == "Team" else col_b:
                user_input[cat] = st.selectbox(cat, options)

    st.divider()
    predict_btn = st.button("⚡  Run Prediction", use_container_width=False)

    if predict_btn:
        try:
            # Build a one-row dataframe matching training features
            input_df = pd.DataFrame([user_input])

            # One-hot encode
            if "Team" in input_df.columns or "Shift" in input_df.columns:
                enc_input = pd.get_dummies(input_df[["Team","Shift"]], drop_first=True, dtype=int)
                input_df  = pd.concat([input_df.drop(["Team","Shift","Employee_ID"], axis=1, errors="ignore"), enc_input], axis=1)
            else:
                input_df.drop(["Employee_ID"], axis=1, errors="ignore", inplace=True)

            # Align with training features
            for f in results["features"]:
                if f not in input_df.columns:
                    input_df[f] = 0
            input_df = input_df[results["features"]]

            scaled = results["scaler"].transform(input_df)
            pred   = results["model"].predict(scaled)[0]
            proba  = results["model"].predict_proba(scaled)[0]

            label      = "Met" if pred == 1 else "Breached"
            conf       = proba[int(pred)] * 100
            style_cls  = "pred-met" if pred == 1 else "pred-breach"
            icon       = "✅" if pred == 1 else "⚠️"

            st.markdown(f"""
            <div class="pred-box">
              <div class="pred-title">Prediction Result</div>
              <div class="pred-result {style_cls}">{icon} SLA {label}</div>
              <p style="color:{MUTED};margin-top:0.5rem;font-size:0.85rem">
                Model confidence: <b style="color:{TEXT}">{conf:.1f}%</b>
              </p>
            </div>""", unsafe_allow_html=True)

            # Probability bar
            fig, ax = plt.subplots(figsize=(7, 1.2))
            ax.barh([""], [proba[1]], color=ACCENT, edgecolor="none", height=0.5, label="Met")
            ax.barh([""], [proba[0]], left=[proba[1]], color=ACCENT2, edgecolor="none", height=0.5, label="Breached")
            ax.text(proba[1]/2, 0, f"Met {proba[1]*100:.1f}%",   va="center", ha="center", fontsize=9, fontweight="bold", color=BG_DARK)
            ax.text(proba[1]+proba[0]/2, 0, f"Breached {proba[0]*100:.1f}%", va="center", ha="center", fontsize=9, fontweight="bold", color=BG_DARK)
            ax.set_xlim(0,1); ax.spines[:].set_visible(False); ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
