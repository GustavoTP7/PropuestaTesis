import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

st.set_page_config(
    page_title="Plataforma Geometalúrgica Flexible", page_icon="⚒️", layout="wide"
)

st.title("⚒️ Gemelo Digital Geometalúrgico & Prescriptivo (Multi-Fuente)")
st.markdown(
    "Carga tus exportaciones de **MineSight / MinePlan**, archivos `.csv` o `.xlsx` sin importar la estructura de columnas."
)

# ----------------------------------------------------
# 1. CARGA Y MAPEO DE DATOS
# ----------------------------------------------------
st.sidebar.header("📁 Fuente de Datos")
uploaded_file = st.sidebar.file_uploader(
    "Cargar Plan/Modelo de Bloques (Excel/CSV)", type=["csv", "xlsx"]
)


@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


if uploaded_file is not None:
    raw_df = load_data(uploaded_file)
    st.sidebar.success(
        f"Archivo cargado: {raw_df.shape[0]} filas, {raw_df.shape[1]} columnas"
    )
else:
    # Dataset por defecto tipo MineSight (Muestra)
    raw_df = pd.DataFrame({
        "ITEM_ID": [f"BLK_{i:04d}" for i in range(1, 101)],
        "BENCH": np.random.choice([3570, 3585, 3600], 100),
        "DOM_GEOMET": np.random.choice(["SSA1", "SSB1", "HYB1", "MIX"], 100),
        "TONS_KT": np.random.uniform(5, 50, 100),
        "CU_TOTAL": np.random.uniform(0.2, 2.2, 100),
        "CU_SOL": np.random.uniform(0.01, 0.4, 100),
        "CU_CN": np.random.uniform(0.05, 1.8, 100),
        "FE_PCT": np.random.uniform(1.0, 5.0, 100),
        "PY_PCT": np.random.uniform(0.5, 6.0, 100),
        "SPI_MIN": np.random.uniform(40, 180, 100),
        "BWI_KWH": np.random.uniform(8.0, 20.0, 100),
        "AXB_VAL": np.random.uniform(25, 120, 100),
    })
    st.info(
        "💡 Usando datos sintéticos de prueba. Carga tu propio archivo desde la barra lateral."
    )

# Mapeo Interactivo de Columnas
with st.expander("🛠️ Mapeo Configurable de Columnas (Adaptador MineSight)", expanded=True):
    cols = ["N/A"] + list(raw_df.columns)

    c1, c2, c3, c4 = st.columns(4)
    col_ton = c1.selectbox(
        "Tonelaje / Volumen",
        cols,
        index=cols.index("TONS_KT") if "TONS_KT" in cols else 0,
    )
    col_cut = c2.selectbox(
        "Ley Cobre Total (%CuT)",
        cols,
        index=cols.index("CU_TOTAL") if "CU_TOTAL" in cols else 0,
    )
    col_cus = c3.selectbox(
        "Cobre Soluble Ácido (%CuS)",
        cols,
        index=cols.index("CU_SOL") if "CU_SOL" in cols else 0,
    )
    col_cucn = c4.selectbox(
        "Cobre Soluble Cianuro (%CuCN)",
        cols,
        index=cols.index("CU_CN") if "CU_CN" in cols else 0,
    )

    c5, c6, c7, c8 = st.columns(4)
    col_fe = c5.selectbox(
        "Hierro (%Fe)",
        cols,
        index=cols.index("FE_PCT") if "FE_PCT" in cols else 0,
    )
    col_spi = c6.selectbox(
        "Índice SPI",
        cols,
        index=cols.index("SPI_MIN") if "SPI_MIN" in cols else 0,
    )
    col_bwi = c7.selectbox(
        "Bond Work Index (BWi)",
        cols,
        index=cols.index("BWI_KWH") if "BWI_KWH" in cols else 0,
    )
    col_axb = c8.selectbox(
        "Parámetro AxB",
        cols,
        index=cols.index("AXB_VAL") if "AXB_VAL" in cols else 0,
    )

    col_dom = st.selectbox(
        "Dominio Geometalúrgico / Alteración",
        cols,
        index=cols.index("DOM_GEOMET") if "DOM_GEOMET" in cols else 0,
    )

# ----------------------------------------------------
# 2. FEATURE ENGINEERING DINÁMICO
# ----------------------------------------------------
df = raw_df.copy()

# Cálculo de ratios en caso existan los campos seleccionados
if col_cut != "N/A" and col_cus != "N/A":
    df["RATIO_CUS_CUT"] = df[col_cus] / np.where(df[col_cut] == 0, 0.001, df[col_cut])

if col_cut != "N/A" and col_cucn != "N/A":
    df["RATIO_CUCN_CUT"] = df[col_cucn] / np.where(df[col_cut] == 0, 0.001, df[col_cut])

if col_spi != "N/A" and col_bwi != "N/A":
    df["INDEX_COMMINUTION"] = df[col_spi] * df[col_bwi] / 100.0

# ----------------------------------------------------
# 3. INTERFAZ MULTI-TAB
# ----------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 1. Exploración & Mezclas (Blending)",
    "🎯 2. Optimización Prescriptiva Custom",
    "🤖 3. Auto-ML & Importancia de Variables",
])

with tab1:
    st.subheader("Análisis Dinámico de la Alimentación")

    if col_dom != "N/A":
        fig_dom = px.histogram(
            df,
            x=col_dom,
            y=col_ton if col_ton != "N/A" else None,
            histfunc="sum",
            title="Distribución de Tonelaje por Dominio Geometalúrgico",
            color=col_dom,
        )
        st.plotly_chart(fig_dom, use_container_width=True)

    st.markdown("### Vista Previa de Datos Procesados")
    st.dataframe(df.head(10), use_container_width=True)

with tab2:
    st.subheader("Configuración del Motor Prescriptivo")
    st.markdown(
        "Ajusta los parámetros operativos requeridos por la planta para el bloque seleccionado."
    )

    col_p1, col_p2, col_p3 = st.columns(3)
    tph_target = col_p1.number_input(
        "Tratamiento Objetivo (TPH)",
        value=3500,
        min_value=1000,
        max_value=6000,
    )
    p80_target = col_p2.slider(
        "P80 Molienda (µm)",
        min_value=90,
        max_value=200,
        value=135,
    )
    ph_target = col_p3.slider(
        "pH Flotación Rougher",
        min_value=9.0,
        max_value=12.0,
        value=10.5,
        step=0.1,
    )

    st.success(
        "El motor procesará dinámicamente las variables mapeadas en el paso 1."
    )

with tab3:
    st.subheader("Importancia de Atributos según Variables Disponibles")

    # Identificar únicamente columnas numéricas elegidas
    numeric_features = [
        c
        for c in [
            col_cut,
            col_cus,
            col_cucn,
            col_fe,
            col_spi,
            col_bwi,
            col_axb,
            "RATIO_CUS_CUT",
            "RATIO_CUCN_CUT",
            "INDEX_COMMINUTION",
        ]
        if c in df.columns and c != "N/A"
    ]

    if len(numeric_features) > 1:
        st.write(f"Variables identificadas para el modelo: `{numeric_features}`")
        # Generación de variable sintética de prueba para entrenamiento dinámico
        y_simulated = (
            80
            + (df[col_cut] * 5 if col_cut != "N/A" else 0)
            - (df[col_spi] * 0.05 if col_spi != "N/A" else 0)
        )

        rf = ExtraTreesRegressor(n_estimators=100, random_state=42)
        rf.fit(df[numeric_features].fillna(0), y_simulated)

        imp_df = (
            pd.DataFrame({
                "Variable": numeric_features,
                "Importancia": rf.feature_importances_,
            })
            .sort_values("Importancia", ascending=True)
        )

        fig_imp = px.bar(
            imp_df,
            x="Importancia",
            y="Variable",
            orientation="h",
            title="Sensibilidad Geometalúrgica Dinámica",
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning(
            "Mapea más de una variable numérica en el paso 1 para activar el análisis de Machine Learning."
        )
