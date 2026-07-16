import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# ──────────────────────────────────────────────────────────────────────────────
# 1) Clase custom del pipeline
# ──────────────────────────────────────────────────────────────────────────────
class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """Target Encoding con suavizado, compatible con Pipeline de scikit-learn."""

    def __init__(self, smoothing=20.0):
        self.smoothing = smoothing

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.cols_ = list(X.columns)
        y = np.asarray(y)
        self.prior_ = y.mean()
        self.maps_ = {}
        for c in self.cols_:
            t = pd.DataFrame({"c": X[c].astype("object"), "y": y})
            agg = t.groupby("c")["y"].agg(["mean", "count"])
            sm = (agg["mean"] * agg["count"] + self.prior_ * self.smoothing) / \
                 (agg["count"] + self.smoothing)
            self.maps_[c] = sm.to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = np.zeros((len(X), len(self.cols_)))
        for i, c in enumerate(self.cols_):
            out[:, i] = X[c].astype("object").map(self.maps_[c]).fillna(self.prior_).values
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"te_{c}" for c in self.cols_])


import __main__
__main__.SmoothedTargetEncoder = SmoothedTargetEncoder

# ──────────────────────────────────────────────────────────────────────────────
# 2) Rutas y carga cacheada de artefactos
#    - cache_resource: objetos no serializables (el pipeline) → 1 vez por servidor
#    - cache_data:     datos tabulares (CSVs del dashboard)   → memoizados
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
RUTA_ARTEFACTO = os.path.join(BASE_DIR, "modelo_porto_seguro.pkl")
DASH_DIR       = os.path.join(BASE_DIR, "dashboard_data")

PALETA = {"LightGBM": "#1f77b4", "Random Forest": "#2ca02c", "Regresión Logística": "#d62728"}


@st.cache_resource(show_spinner="Cargando modelo entrenado...")
def cargar_artefacto(ruta: str) -> dict:
    """Bundle {pipeline, metadatos} generado por la celda de serialización."""
    return joblib.load(ruta)


@st.cache_data(show_spinner=False)
def leer_csv(nombre: str) -> pd.DataFrame:
    """Lee un CSV estático del dashboard (precomputado en el notebook)."""
    return pd.read_csv(os.path.join(DASH_DIR, nombre))


# ──────────────────────────────────────────────────────────────────────────────
# 3) Lógica de inferencia (Pestaña 2)
# ──────────────────────────────────────────────────────────────────────────────
def construir_fila(art: dict, valores_usuario: dict) -> pd.DataFrame:
    """Fila completa de entrada: asegurado promedio + inputs del usuario,
    recalculando las variables derivadas fe_* (idéntico a la Celda 7bis)."""
    fila = dict(art["valores_base"])
    fila.update(valores_usuario)
    if {"ps_car_13", "ps_reg_03"} <= fila.keys():
        fila["fe_car13_reg03_mult"]  = fila["ps_car_13"] * fila["ps_reg_03"]
        fila["fe_car13_reg03_ratio"] = fila["ps_car_13"] / (abs(fila["ps_reg_03"]) + 1e-3)
    if "ps_car_13" in fila:
        fila["fe_car13_sq"] = fila["ps_car_13"] ** 2
    return pd.DataFrame([fila])[art["features"]]


def predecir(art: dict, valores_usuario: dict) -> tuple[float, int]:
    """(probabilidad de siniestro, decil de riesgo 1-10 en la escala OOF)."""
    X_new = construir_fila(art, valores_usuario)
    proba = float(art["pipeline"].predict_proba(X_new)[0, 1])
    decil = int(np.searchsorted(art["cortes_deciles"], proba) + 1)
    return proba, decil


# ──────────────────────────────────────────────────────────────────────────────
# 4) Configuración general
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Porto Seguro · Riesgo de Siniestro", layout="wide")

st.title("Porto Seguro — Safe Driver Prediction")
st.caption("Pipeline CRISP-DM completo · Modelo ganador: **LightGBM optimizado (Optuna)** · "
           "Métrica principal: **Gini Normalizado** sobre predicciones *out-of-fold*")

tab_dash, tab_sim = st.tabs(["Dashboard de Modelo y Negocio", "Simulador Predictivo"])

# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑA 1 — DASHBOARD (lee CSVs estáticos; no recalcula)
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    try:
        kpis    = leer_csv("kpis.csv").set_index("KPI")["Valor"]
        roc     = leer_csv("roc_curvas.csv")
        giniauc = leer_csv("gini_auc.csv")
        featimp = leer_csv("feature_importance.csv")
        deciles = leer_csv("deciles.csv")
        perfil  = leer_csv("perfil_riesgo.csv")
    except FileNotFoundError as e:
        st.error(
            f"Falta un archivo del dashboard: `{e.filename}`. Ejecuta la celda "
            "**Fase 6 (b)** del notebook para generar `dashboard_data/` y súbela "
            "al repositorio junto a `app.py`."
        )
        st.stop()

    # ── KPIs superiores ───────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pólizas evaluadas",     f"{int(kpis['Total_Polizas']):,}")
    k2.metric("Tasa de reclamos",      f"{kpis['Tasa_Reclamos_Pct']:.2f} %",
              help="Porcentaje real de siniestros en el dataset (clase positiva).")
    k3.metric("Gini Normalizado (LightGBM)", f"{kpis['Gini_LightGBM']:.4f}",
              help="Calculado sobre predicciones out-of-fold de la validación cruzada.")
    k4.metric("Siniestros capturados en decil 10", f"{kpis['Concentracion_D10_Pct']:.1f} %",
              help="Proporción de todos los reclamos reales que cae en el 10% más riesgoso.")

    st.divider()

    # ── Gráficos matemáticos: ROC + Gini/AUC ──────────────────────────────────
    st.subheader("Evaluación matemática de los modelos")
    c_roc, c_gini = st.columns(2)

    with c_roc:
        fig_roc = go.Figure()
        for modelo, g in roc.groupby("Modelo"):
            fig_roc.add_trace(go.Scatter(x=g["FPR"], y=g["TPR"], name=modelo,
                                         mode="lines",
                                         line=dict(color=PALETA.get(modelo))))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Azar (AUC=0.5)",
                                     mode="lines",
                                     line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(title="Curvas ROC (out-of-fold)",
                              xaxis_title="Tasa de falsos positivos (FPR)",
                              yaxis_title="Tasa de verdaderos positivos (TPR)",
                              legend=dict(orientation="h", y=-0.25), height=430)
        st.plotly_chart(fig_roc, width="stretch")

    with c_gini:
        g_long = giniauc.melt(id_vars="Modelo", value_vars=["Gini", "AUC"],
                              var_name="Métrica", value_name="Valor")
        fig_gini = px.bar(g_long, x="Modelo", y="Valor", color="Métrica",
                          barmode="group", text_auto=".4f",
                          title="Gini Normalizado y AUC por modelo",
                          color_discrete_sequence=["#1f77b4", "#9ecae1"])
        fig_gini.update_layout(yaxis_range=[0, 1],
                               legend=dict(orientation="h", y=-0.25), height=430)
        st.plotly_chart(fig_gini, width="stretch")

    st.divider()

    # ── Gráficos de negocio: importancia + deciles ────────────────────────────
    st.subheader("Lectura de negocio del modelo")
    c_imp, c_dec = st.columns(2)

    with c_imp:
        fi = featimp.sort_values("Peso_SHAP_Pct")     # ascendente → barra mayor arriba
        fig_imp = px.bar(fi, x="Peso_SHAP_Pct", y="Atributo", orientation="h",
                         text_auto=".1f",
                         title="Top 10 variables más importantes (SHAP)",
                         labels={"Peso_SHAP_Pct": "Peso SHAP (%)", "Atributo": ""})
        fig_imp.update_traces(marker_color="#1f77b4")
        fig_imp.update_layout(height=430)
        st.plotly_chart(fig_imp, width="stretch")
        st.caption("Importancia global por valores de Shapley (|SHAP| medio, normalizado a 100%), "
                   "con dummies One-Hot agregadas a su variable original.")

    with c_dec:
        fig_dec = go.Figure()
        fig_dec.add_trace(go.Bar(x=deciles["Decil"], y=deciles["Prob_Media_Pct"],
                                 name="Prob. media predicha (%)", marker_color="#1f77b4"))
        fig_dec.add_trace(go.Scatter(x=deciles["Decil"], y=deciles["Tasa_Real_Pct"],
                                     name="Tasa real de reclamos (%)",
                                     mode="lines+markers", line=dict(color="#d62728")))
        fig_dec.update_layout(title="Deciles de riesgo: predicción vs. realidad",
                              xaxis=dict(title="Decil (10 = mayor riesgo)", dtick=1),
                              yaxis_title="%",
                              legend=dict(orientation="h", y=-0.25), height=430)
        st.plotly_chart(fig_dec, width="stretch")
        st.caption("La monotonicidad (la tasa real crece con el decil predicho) evidencia "
                   "la calidad del ordenamiento, que es justo lo que mide el Gini.")

    # ── Tabla perfil de riesgo ────────────────────────────────────────────────
    st.subheader("Perfil del cliente: Alto riesgo (P90+) vs. Bajo riesgo (P0–60)")
    st.dataframe(
        perfil.style.format({"Riesgo_Bajo": "{:.4f}", "Riesgo_Alto": "{:.4f}",
                             "Ratio_Alto_vs_Bajo": "{:.2f}×"})
              .background_gradient(subset=["Ratio_Alto_vs_Bajo"], cmap="Reds"),
        width="stretch", hide_index=True,
    )
    st.caption("Medias por segmento según la probabilidad OOF de LightGBM. El ratio resalta "
               "qué características separan más a los segmentos (incluida la tasa real de reclamos).")

# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑA 2 — SIMULADOR PREDICTIVO (inferencia en vivo con el .pkl)
# ══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    try:
        art = cargar_artefacto(RUTA_ARTEFACTO)
    except FileNotFoundError:
        st.error(f"No se encontró `{os.path.basename(RUTA_ARTEFACTO)}`. Genera el artefacto con la "
                 "celda de serialización (Fase 6) del notebook y súbelo junto a `app.py`.")
        st.stop()

    # ── Sidebar: inputs de las variables Top-SHAP ─────────────────────────────
    st.sidebar.header("Variables del asegurado")
    st.sidebar.caption("Top de variables por importancia SHAP (Fase 5 · Evaluación). "
                       "Aplican a la pestaña **Simulador Predictivo**.")

    valores_usuario = {}
    for f in art["ui_features"]:
        if f["entera"]:
            valores_usuario[f["nombre"]] = float(st.sidebar.slider(
                f["nombre"], min_value=int(round(f["min"])), max_value=int(round(f["max"])),
                value=int(round(f["default"])), step=1))
        else:
            valores_usuario[f["nombre"]] = float(st.sidebar.slider(
                f["nombre"], min_value=round(f["min"], 4), max_value=round(f["max"], 4),
                value=round(f["default"], 4), step=round(f["step"], 4), format="%.4f"))

    if st.sidebar.button("Restablecer al asegurado promedio"):
        st.rerun()

    st.markdown("Ajusta las variables en la **barra lateral**; los demás atributos se completan "
                "con el perfil del *asegurado promedio* aprendido en entrenamiento (mediana/moda). "
                "Las variables derivadas `fe_*` se recalculan automáticamente.")

    # ── Botón central de predicción ───────────────────────────────────────────
    if st.button("🔮 Calcular riesgo de siniestro", type="primary", width="stretch"):
        proba, decil = predecir(art, valores_usuario)
        tasa_base = art["tasa_base_pct"]

        if decil >= 9:
            nivel, color = "ALTO", "🔴"
        elif decil >= 7:
            nivel, color = "MEDIO", "🟠"
        else:
            nivel, color = "BAJO", "🟢"

        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilidad de siniestro", f"{proba * 100:.2f} %",
                  delta=f"{proba * 100 - tasa_base:+.2f} pp vs. tasa base",
                  delta_color="inverse")
        c2.metric("Decil de riesgo sugerido", f"{decil} / 10")
        c3.metric("Nivel", f"{color} {nivel}")

        st.progress(min(decil / 10, 1.0),
                    text=f"Posición en la escala de riesgo (decil {decil} de 10)")

        with st.expander("🔎 Ver fila completa enviada al modelo"):
            st.dataframe(construir_fila(art, valores_usuario).T.rename(columns={0: "valor"}),
                         width="stretch")

        st.caption(f"Tasa base de siniestros: {tasa_base:.2f} %. El decil se calcula con los "
                   "cortes de las probabilidades out-of-fold de la validación cruzada "
                   "(decil 10 = mayor riesgo), la misma escala del Dashboard.")
    else:
        st.info("Configura el perfil en la barra lateral y pulsa **Calcular riesgo de siniestro**.")

    # ── Trazabilidad ──────────────────────────────────────────────────────────
    with st.expander("Detalles técnicos del modelo desplegado"):
        st.write("**Versiones del entorno de entrenamiento** (deben coincidir con `requirements.txt`):")
        st.json(art["versiones"])
        st.write(f"**Nº de features del pipeline:** {len(art['features'])}")
        st.write("**Variables expuestas en la interfaz:**",
                 ", ".join(f["nombre"] for f in art["ui_features"]))
