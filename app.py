import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostRegressor

# Page configuration
st.set_page_config(page_title="Plant-Based Biomass Coarse Aggregate Concrete Strength Predictor", layout="wide")
st.title("🌱 Plant-Based Biomass Coarse Aggregate Concrete Compressive Strength Predictor")
st.markdown("Powered by **Bayesian-optimized CatBoost** – interpretable design tool with SHAP analysis and mix optimization recommendations.")

# ---------- Load model ----------
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("best_model_cb.cbm")
    return model

try:
    model = load_model()
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ---------- Default values for reset ----------
defaults = {
    "cement": 400.0,
    "water": 170.0,
    "prewater": 50.0,
    "sand": 900.0,
    "nca": 800.0,
    "pbca": 200.0,
    "r": 25.0,
    "sp": 6.0,
    "wb": 10.0,
    "rho_b": 800.0,
    "dbmax": 10.0,
    "curing_age": 28,
    "slump": 150
}

# Initialize session state keys if not exist
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------- Sidebar input controls ----------
st.sidebar.header("📊 Input Mix Parameters")
st.sidebar.markdown("💡 **Note**: R is the **volume-based** replacement ratio by volume of coarse aggregate.")

# Reset button
if st.sidebar.button("🔄 Reset to Defaults"):
    for key, val in defaults.items():
        st.session_state[key] = val
    st.rerun()

# Main materials
cement = st.sidebar.number_input("Cement (kg/m³)", min_value=0.0, max_value=800.0, value=st.session_state.cement, step=5.0, key="cement")
water = st.sidebar.number_input("Water (kg/m³)", min_value=0.0, max_value=300.0, value=st.session_state.water, step=5.0, key="water")
prewater = st.sidebar.number_input("Pre-wetted water (kg/m³)", min_value=0.0, max_value=150.0, value=st.session_state.prewater, step=5.0, key="prewater")
sand = st.sidebar.number_input("Sand (kg/m³)", min_value=0.0, max_value=1500.0, value=st.session_state.sand, step=10.0, key="sand")
nca = st.sidebar.number_input("Natural coarse aggregate (kg/m³)", min_value=0.0, max_value=1600.0, value=st.session_state.nca, step=10.0, key="nca")
pbca = st.sidebar.number_input("Plant-based biomass coarse aggregate (kg/m³)", min_value=0.0, max_value=800.0, value=st.session_state.pbca, step=5.0, key="pbca")

r = st.sidebar.number_input(
    "Replacement ratio R (%)",
    min_value=0.0,
    max_value=100.0,
    value=st.session_state.r,
    step=1.0,
    help="Volume-based replacement ratio: V_PBCA / (V_PBCA + V_NCA).",
    key="r"
)

sp = st.sidebar.number_input("Superplasticizer (kg/m³)", min_value=0.0, max_value=30.0, value=st.session_state.sp, step=0.5, key="sp")

# Plant-based biomass aggregate properties
wb = st.sidebar.number_input(
    "Water absorption of plant-based biomass coarse aggregate (%)",
    min_value=0.0,
    max_value=50.0,
    value=st.session_state.wb,
    step=1.0,
    key="wb"
)

rho_b = st.sidebar.number_input(
    "Bulk density of plant-based biomass coarse aggregate (kg/m³)",
    min_value=300.0,
    max_value=1300.0,
    value=st.session_state.rho_b,
    step=10.0,
    key="rho_b"
)

dbmax = st.sidebar.number_input(
    "Maximum particle size of plant-based biomass coarse aggregate (mm)",
    min_value=1.0,
    max_value=50.0,
    value=st.session_state.dbmax,
    step=1.0,
    key="dbmax"
)

# Process parameters
curing_age = st.sidebar.number_input("Curing age (days)", min_value=1, max_value=365, value=st.session_state.curing_age, step=1, key="curing_age")
slump = st.sidebar.number_input("Slump (mm)", min_value=0, max_value=300, value=st.session_state.slump, step=5, key="slump")

# Derived features
wc = water / cement if cement > 0 else 0.0
cs_ratio = cement / sand if sand > 0 else 0.0
cpb_ratio = cement / pbca if pbca > 0 else 0.0

# ---------- Build input dictionary ----------
input_dict = {
    "Curing age": curing_age,
    "Cement": cement,
    "Water": water,
    "Pre-wetted water": prewater,
    "Sand": sand,
    "NCA": nca,
    "PBCA": pbca,
    "R": r,
    "SP": sp,
    "Water/Cement": wc,
    "Cement/Sand": cs_ratio,
    "Cement/PBCA": cpb_ratio,
    "WB": wb,
    "ρB": rho_b,
    "DB-MAX": dbmax,
    "Slump": slump
}

# ---------- Feature order ----------
feature_order = [
    "Curing age",
    "Cement",
    "Water",
    "Pre-wetted water",
    "Sand",
    "NCA",
    "PBCA",
    "R",
    "SP",
    "Water/Cement",
    "Cement/Sand",
    "Cement/PBCA",
    "WB",
    "ρB",
    "DB-MAX",
    "Slump"
]

# ---------- Prediction button ----------
if st.sidebar.button("🚀 Predict Strength", type="primary"):
    input_vector = [input_dict[feat] for feat in feature_order]
    prediction = model.predict([input_vector])[0]

    st.subheader("📈 Prediction Result")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="**Predicted Compressive Strength (CS)**",
            value=f"{prediction:.2f} MPa",
            delta=f"{prediction - 23:.2f} MPa vs baseline (23 MPa)"
        )

    with col2:
        st.info("💡 Prediction based on your input parameters – useful for mix design optimization.")

    # SHAP optional
    try:
        # If you have a saved TreeExplainer, uncomment below
        # explainer = joblib.load("explainer.pkl")
        # shap_values = explainer.shap_values([input_vector])
        # fig, ax = plt.subplots()
        # shap.waterfall_plot(
        #     shap.Explanation(
        #         values=shap_values[0],
        #         base_values=explainer.expected_value,
        #         data=np.array(input_vector),
        #         feature_names=feature_order
        #     ),
        #     show=False
        # )
        # st.pyplot(fig)

        st.markdown("📖 **SHAP model explanation** – To enable, save a TreeExplainer object as 'explainer.pkl'.")
    except:
        st.warning("⚠️ SHAP explainer not loaded, skipping explanation chart.")

    # Mix design recommendations
    st.subheader("🔧 Mix Design Recommendations")
    recs = []

    if wc < 0.40:
        recs.append("✅ **Water/cement ratio in optimal range (<0.40)**, favorable for high strength.")
    elif 0.40 <= wc <= 0.45:
        recs.append("⚠️ **Water/cement ratio at transition zone (0.40–0.45)** – strength starts to decrease. Consider reducing to <0.40.")
    else:
        recs.append("❌ **Water/cement ratio too high (>0.45)** – strength drops sharply. Keep W/C within 0.30–0.40.")

    if wc < 0.40 and curing_age < 28:
        recs.append("ℹ️ With low W/C, extending curing age (≥28 days) can further improve strength.")
    elif wc > 0.45:
        recs.append("⚠️ With high W/C, prolonged curing has limited effect. Reduce W/C first.")
    elif curing_age >= 28:
        recs.append("✅ Curing age sufficient; combined with low W/C leads to good strength.")

    water_ok = 160 <= water <= 180
    sp_ok = 5 <= sp <= 7.5
    slump_ok = 150 <= slump <= 220

    if water_ok and sp_ok and slump_ok:
        recs.append("✅ **Water (160–180), SP (5–7.5) and slump (150–220) are in synergistic optimal zone.**")
    else:
        if not water_ok:
            recs.append("⚠️ Water content should be 160–180 kg/m³.")
        if not sp_ok:
            recs.append("⚠️ SP dosage recommended 5–7.5 kg/m³.")
        if not slump_ok:
            recs.append("⚠️ Slump recommended 150–220 mm.")
        if water > 200 or sp > 10:
            recs.append("❌ Water >200 kg/m³ or SP >10 kg/m³ enters rapid strength degradation zone.")

    if r <= 20:
        recs.append("✅ Replacement ratio R ≤20% – stable strength.")
    elif 20 < r <= 30:
        recs.append("⚠️ R in critical transition zone (20–30%) – strength highly sensitive.")
    else:
        recs.append("❌ **R >30%** – strength fully governed by plant-based biomass aggregate. Compensatory measures required:")

        if cement < 450:
            recs.append("   • Increase cement content (≥450 kg/m³).")
        if water > 180:
            recs.append("   • Lower water to 160–180 kg/m³.")
        if cs_ratio < 0.7:
            recs.append("   • Increase sand ratio (Cement/Sand ≈0.7–0.9).")
        if rho_b < 800:
            recs.append("   • Plant-based biomass aggregate bulk density should be ≥800 kg/m³.")
        if wb > 10:
            recs.append("   • Plant-based biomass aggregate water absorption ≤10% recommended.")
        if dbmax > 10:
            recs.append("   • Maximum particle size should be 5–10 mm.")

    if rho_b < 800:
        recs.append("⚠️ Plant-based biomass aggregate bulk density low (<800 kg/m³). Choose higher-density aggregate.")

    if wb > 15:
        recs.append("⚠️ Plant-based biomass aggregate water absorption high (>15%). Pre-treatment advised.")

    if dbmax > 15:
        recs.append("⚠️ Plant-based biomass aggregate maximum particle size too large (>15 mm). Recommended 5–10 mm.")

    if wc > 0.45 and cs_ratio < 0.6:
        recs.append("ℹ️ Under high W/C, increasing sand ratio can partly compensate density loss.")

    for rec in recs:
        st.markdown(rec)

    with st.expander("📋 View input parameters"):
        input_df = pd.DataFrame([input_dict])
        st.dataframe(input_df.T, use_container_width=True)

# ---------- Footer ----------
st.markdown("---")
st.caption("⚠️ This tool is for research purposes only. Always validate predictions with laboratory tests.")
