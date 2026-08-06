from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Prediksi Kualitas Air",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "model_rf_{dataset_label}.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def label_color(label: str) -> str:
    normalized = str(label).lower()
    if "aman" in normalized and "tidak" not in normalized:
        return "#15803d"
    if "tidak" in normalized:
        return "#b91c1c"
    return "#b45309"


def label_description(label: str) -> str:
    normalized = str(label).lower()
    if "aman" in normalized and "tidak" not in normalized:
        return "Parameter air berada pada kategori aman menurut model."
    if "tidak" in normalized:
        return "Parameter air berada pada kategori tidak aman menurut model."
    return "Hasil berada pada kategori perlu perhatian dan sebaiknya ditinjau lebih lanjut."


# Styling tambahan agar tampilan konsisten dan mudah dibaca.
st.markdown(
    """
    <style>
    .main { background: #f7fafc; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }
    .hero { padding: 1.5rem 1.75rem; border-radius: 20px; background: linear-gradient(135deg, #0f766e 0%, #155e75 100%); color: white; margin-bottom: 1.5rem; }
    .hero h1 { margin: 0 0 .35rem 0; font-size: 2.2rem; }
    .hero p { margin: 0; opacity: .9; font-size: 1.05rem; }
    .result-card { padding: 1.25rem 1.4rem; border-radius: 16px; border: 1px solid #dbe4ea; background: white; box-shadow: 0 8px 24px rgba(15, 118, 110, .08); }
    .result-label { color: #64748b; font-size: .9rem; margin-bottom: .35rem; }
    .result-value { font-size: 2rem; font-weight: 750; margin: 0; }
    .muted { color: #64748b; font-size: .92rem; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #e2e8f0; padding: 1rem; border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Prediksi Kualitas Air</h1>
      <p>Evaluasi kualitas air menggunakan model Random Forest berdasarkan pH, turbidity, dan TDS.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_model()
except Exception as error:
    st.error("Model belum dapat dimuat. Pastikan file model berada satu folder dengan app.py.")
    st.exception(error)
    st.stop()

classes = list(getattr(model, "classes_", []))
feature_names = list(getattr(model, "feature_names_in_", ["pH", "Turbidity", "TDS"]))

with st.sidebar:
    st.header("Parameter Air")
    st.caption("Masukkan nilai pengukuran laboratorium atau sensor.")
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1, format="%.2f", help="Rentang umum skala pH adalah 0 sampai 14.")
    turbidity = st.number_input("Turbidity / Kekeruhan", min_value=0.0, value=1.0, step=0.1, format="%.2f", help="Gunakan satuan yang sama seperti saat model dilatih.")
    tds = st.number_input("TDS", min_value=0.0, value=100.0, step=1.0, format="%.2f", help="Gunakan satuan yang sama seperti saat model dilatih.")
    st.divider()
    predict_clicked = st.button("Prediksi Sekarang", type="primary", use_container_width=True)
    st.caption("Catatan: hasil mengikuti pola data dan definisi label pada saat model dilatih.")

input_data = pd.DataFrame([[ph, turbidity, tds]], columns=feature_names)

col_a, col_b, col_c = st.columns(3)
col_a.metric("pH", f"{ph:.2f}")
col_b.metric("Turbidity", f"{turbidity:.2f}")
col_c.metric("TDS", f"{tds:.2f}")

st.subheader("Hasil Prediksi")

# Prediksi otomatis pada pemuatan awal dan setiap perubahan nilai.
if predict_clicked or "prediction" not in st.session_state:
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0] if hasattr(model, "predict_proba") else None
    st.session_state.prediction = prediction
    st.session_state.probabilities = probabilities

prediction = st.session_state.prediction
probabilities = st.session_state.probabilities
confidence = float(max(probabilities)) * 100 if probabilities is not None else None
color = label_color(str(prediction))

a, b = st.columns([1.1, 1.9])
with a:
    confidence_text = f"{confidence:.1f}%" if confidence is not None else "Tidak tersedia"
    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-label">Kategori kualitas air</div>
          <p class="result-value" style="color:{color}">{prediction}</p>
          <p class="muted">{label_description(str(prediction))}</p>
          <hr>
          <div class="result-label">Tingkat keyakinan model</div>
          <p class="result-value">{confidence_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with b:
    st.markdown("#### Probabilitas setiap kategori")
    if probabilities is not None and classes:
        probability_df = pd.DataFrame({"Kategori": classes, "Probabilitas": probabilities})
        probability_df["Persentase"] = probability_df["Probabilitas"] * 100
        st.bar_chart(probability_df.set_index("Kategori")["Persentase"], y_label="Probabilitas (%)", x_label="Kategori")
    else:
        st.info("Model ini tidak menyediakan probabilitas kelas.")

st.divider()
left, right = st.columns([1.2, 1.8])
with left:
    st.subheader("Nilai yang digunakan")
    st.dataframe(input_data, use_container_width=True, hide_index=True)
with right:
    st.subheader("Kontribusi fitur model")
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({"Fitur": feature_names, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Fitur"), y_label="Importance")
    else:
        st.info("Feature importance tidak tersedia pada model ini.")

with st.expander("Detail teknis model"):
    st.write(f"**Tipe model:** `{type(model).__name__}`")
    st.write(f"**Jumlah estimator:** `{getattr(model, 'n_estimators', 'N/A')}`")
    st.write(f"**Fitur:** `{', '.join(feature_names)}`")
    st.write(f"**Kelas:** `{', '.join(map(str, classes))}`")
