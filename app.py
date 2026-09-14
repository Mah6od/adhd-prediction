import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Page config + dark theme
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="ADHD Diagnosis Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
:root {
    --bg: #0e1117;
    --panel: #161b22;
    --panel-2: #1c2128;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #7aa2f7;
    --border: #30363d;
}
html, body, [class*="css"] { color: var(--text); }
.stApp { background-color: var(--bg); }
section[data-testid="stSidebar"] {
    background-color: var(--panel);
    border-right: 1px solid var(--border);
}
div.block-container { padding-top: 2rem; }
h1, h2, h3 { color: var(--text) !important; }
p, span, label, .stMarkdown { color: var(--text); }
.stButton > button {
    background-color: var(--accent);
    color: #0e1117;
    border: none;
    font-weight: 600;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
}
.stButton > button:hover {
    background-color: #9db9fb;
    color: #0e1117;
}
div[data-testid="stMetric"] {
    background-color: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
}
.result-card {
    background-color: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.class-healthy { color: #3fb950; font-weight: 700; }
.class-adhd { color: #f0883e; font-weight: 700; }
.class-other { color: #d29922; font-weight: 700; }
hr { border-color: var(--border); }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

plt.style.use("dark_background")
FIG_BG = "#0e1117"

CLASS_STYLE = {
    0: ("Healthy", "class-healthy"),
    1: ("ADHD", "class-adhd"),
    2: ("Other Dx", "class-other"),
}

# --------------------------------------------------------------------------------------
# Load model bundle
# --------------------------------------------------------------------------------------
BUNDLE_PATH = "models/adhd_model_bundle.joblib"


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


@st.cache_resource
def get_explainer(_model, _background):
    return shap.Explainer(_model.predict_proba, _background)


try:
    bundle = load_bundle(BUNDLE_PATH)
except FileNotFoundError:
    st.error(
        f"Model bundle not found at `{BUNDLE_PATH}`. "
        "Run the notebook's save-model cell first and place the .joblib file "
        "in the same folder as this app."
    )
    st.stop()

model = bundle["model"]
scaler = bundle["scaler"]
winsor_bounds = bundle["winsor_bounds"]
feature_cols = bundle["feature_cols"]
classes = bundle["classes"]
class_names = bundle["class_names"]
model_name = bundle.get("best_tuned_name", type(model).__name__)

# Background sample for SHAP — stash a small slice of scaled training-like data.
# If the bundle doesn't carry one, fall back to synthesizing from winsor bounds.
if "shap_background" in bundle:
    background = bundle["shap_background"]
else:
    rng = np.random.default_rng(42)
    synth_rows = []
    for _ in range(100):
        row = {}
        for col in feature_cols:
            lo, hi = winsor_bounds.get(col, (0, 1))
            row[col] = rng.uniform(lo, hi)
        synth_rows.append(row)
    bg_raw = pd.DataFrame(synth_rows)[feature_cols]
    background = pd.DataFrame(scaler.transform(bg_raw), columns=feature_cols)

explainer = get_explainer(model, background)

# --------------------------------------------------------------------------------------
# Sidebar — feature inputs
# --------------------------------------------------------------------------------------
st.sidebar.title("🧠 Subject Features")
st.sidebar.caption(f"Model in use: **{model_name}**")
st.sidebar.markdown("---")

st.sidebar.subheader("Demographics")
gender = st.sidebar.selectbox("Gender", options=[0.0, 1.0], format_func=lambda x: "Female (0)" if x == 0.0 else "Male (1)")
age = st.sidebar.slider("Age (years)", min_value=6, max_value=40, value=11, step=1)
handedness = st.sidebar.selectbox("Handedness", options=[0.0, 1.0], format_func=lambda x: "Left (0)" if x == 0.0 else "Right (1)")
med_status = st.sidebar.selectbox("Medication Status", options=[1, 2], format_func=lambda x: "Not medicated (1)" if x == 1 else "Medicated (2)")

st.sidebar.subheader("Clinical Scores")
adhd_index = st.sidebar.slider("ADHD Index", min_value=30.0, max_value=90.0, value=55.0, step=0.5)
inattentive = st.sidebar.slider("Inattentive", min_value=30.0, max_value=90.0, value=55.0, step=0.5)
hyper_impulsive = st.sidebar.slider("Hyper/Impulsive", min_value=30.0, max_value=90.0, value=55.0, step=0.5)

st.sidebar.subheader("IQ Scores")
verbal_iq = st.sidebar.slider("Verbal IQ", min_value=60.0, max_value=150.0, value=100.0, step=1.0)
performance_iq = st.sidebar.slider("Performance IQ", min_value=60.0, max_value=150.0, value=100.0, step=1.0)
full4_iq = st.sidebar.slider("Full4 IQ", min_value=60.0, max_value=150.0, value=100.0, step=1.0)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🔍 Predict", use_container_width=True)

# --------------------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------------------
st.title("ADHD Diagnosis Classifier")
st.caption(
    "Enter subject features in the sidebar and click **Predict** to classify into "
    "Healthy / ADHD / Other Diagnosis, with a SHAP explanation of the prediction."
)
st.markdown("---")


def build_input_row():
    return {
        "Gender": gender,
        "Age": age,
        "Handedness": handedness,
        "ADHD Index": adhd_index,
        "Inattentive": inattentive,
        "Hyper_Impulsive": hyper_impulsive,
        "Verbal IQ": verbal_iq,
        "Performance IQ": performance_iq,
        "Full4 IQ": full4_iq,
        "Med Status": med_status,
    }


def preprocess(raw_input: dict):
    df_new = pd.DataFrame([raw_input])[feature_cols]
    for col, (lo, hi) in winsor_bounds.items():
        if col in df_new.columns:
            df_new[col] = df_new[col].clip(lower=lo, upper=hi)
    x_scaled = pd.DataFrame(scaler.transform(df_new), columns=feature_cols)
    return x_scaled


if run_btn:
    raw_input = build_input_row()
    x_scaled = preprocess(raw_input)

    pred = model.predict(x_scaled)[0]
    proba = model.predict_proba(x_scaled)[0]
    pred_label, pred_css = CLASS_STYLE.get(pred, (str(pred), ""))

    # ---- Result card ----
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"""
            <div class="result-card">
                <p style="color:var(--muted); margin-bottom:0.2rem;">Predicted Diagnosis</p>
                <h2 class="{pred_css}" style="margin-top:0;">{pred_label}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for c in classes:
            label, _ = CLASS_STYLE.get(c, (str(c), ""))
            st.metric(label, f"{proba[list(classes).index(c)]:.1%}")

    with col2:
        st.subheader("Class Probabilities")
        prob_labels = [CLASS_STYLE.get(c, (str(c), ""))[0] for c in classes]
        fig_prob, ax_prob = plt.subplots(figsize=(7, 4))
        fig_prob.patch.set_facecolor(FIG_BG)
        ax_prob.set_facecolor(FIG_BG)
        bar_colors = ["#7aa2f7" if c != pred else "#f0883e" for c in classes]
        ax_prob.bar(prob_labels, proba, color=bar_colors)
        ax_prob.set_ylim(0, 1.0)
        ax_prob.set_ylabel("Probability", color="#e6edf3")
        ax_prob.tick_params(colors="#e6edf3")
        for spine in ax_prob.spines.values():
            spine.set_color("#30363d")
        for i, v in enumerate(proba):
            ax_prob.text(i, v + 0.02, f"{v:.1%}", ha="center", color="#e6edf3", fontsize=10)
        st.pyplot(fig_prob, clear_figure=True)

    st.markdown("---")

    # ---- SHAP explanation ----
    st.subheader(f"Why did the model predict '{pred_label}'?")
    with st.spinner("Computing SHAP values..."):
        sv = explainer(x_scaled)
        class_idx = list(classes).index(pred)

        sv_single = shap.Explanation(
            values=sv.values[0, :, class_idx],
            base_values=sv.base_values[0, class_idx],
            data=x_scaled.iloc[0].values,
            feature_names=feature_cols,
        )

    # ---- Plain-language driver callout ----
    contrib_all = pd.Series(sv_single.values, index=feature_cols)
    pushed_for = contrib_all[contrib_all > 0].sort_values(ascending=False)
    pushed_against = contrib_all[contrib_all < 0].sort_values()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Pushed the model TOWARD '{pred_label}':**")
        if len(pushed_for) > 0:
            for feat, val in pushed_for.head(5).items():
                raw_val = raw_input.get(feat)
                st.markdown(f"- 🟢 **{feat}** (value: {raw_val}) — impact +{val:.3f}")
        else:
            st.caption("No features pushed toward this class.")
    with c2:
        st.markdown(f"**Pushed the model AWAY from '{pred_label}':**")
        if len(pushed_against) > 0:
            for feat, val in pushed_against.head(5).items():
                raw_val = raw_input.get(feat)
                st.markdown(f"- 🔴 **{feat}** (value: {raw_val}) — impact {val:.3f}")
        else:
            st.caption("No features pushed away from this class.")

    top_driver = contrib_all.abs().idxmax()
    top_driver_val = contrib_all[top_driver]
    direction = "toward" if top_driver_val > 0 else "away from"
    st.success(
        f"**Biggest single driver:** `{top_driver}` (value: {raw_input.get(top_driver)}), "
        f"pushing the prediction **{direction} '{pred_label}'** "
        f"(SHAP impact: {top_driver_val:+.3f})."
    )

    st.markdown("---")

    tab1, tab2 = st.tabs(["Waterfall (this prediction)", "Feature contributions (bar)"])

    with tab1:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.patch.set_facecolor(FIG_BG)
        ax.set_facecolor(FIG_BG)
        shap.plots.waterfall(sv_single, show=False)
        fig_current = plt.gcf()
        fig_current.patch.set_facecolor(FIG_BG)
        for a in fig_current.get_axes():
            a.set_facecolor(FIG_BG)
            a.tick_params(colors="#e6edf3")
            a.xaxis.label.set_color("#e6edf3")
            a.yaxis.label.set_color("#e6edf3")
        st.pyplot(fig_current, clear_figure=True)

    with tab2:
        contrib = pd.Series(sv_single.values, index=feature_cols).sort_values()
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor(FIG_BG)
        ax2.set_facecolor(FIG_BG)
        colors = ["#f85149" if v < 0 else "#3fb950" for v in contrib.values]
        ax2.barh(contrib.index, contrib.values, color=colors)
        ax2.axvline(0, color="#8b949e", linewidth=0.8)
        ax2.set_xlabel("SHAP value (impact on predicted class)", color="#e6edf3")
        ax2.tick_params(colors="#e6edf3")
        st.pyplot(fig2, clear_figure=True)

    st.caption(
        f"Base value (expected model output for {pred_label}): "
        f"{sv_single.base_values:.3f} → Final prediction driven by the feature "
        f"contributions above."
    )

    with st.expander("Show raw input & preprocessed values"):
        st.write("**Raw input:**")
        st.dataframe(pd.DataFrame([raw_input]), use_container_width=True)
        st.write("**Scaled input (as fed to model):**")
        st.dataframe(x_scaled, use_container_width=True)

else:
    st.info("👈 Set the subject's features in the sidebar, then click **Predict**.")