Aquí tienes una aplicación completa y funcional con **Streamlit** que incluye la interfaz gráfica para subir los datos del turno (o usar los datos por defecto de tu imagen), calcular las métricas ponderadas, predecir la recuperación con XGBoost y mostrar el panel de control con los setpoints operacionales.

Para ejecutarlo localmente:

1. Instala las librerías: `pip install streamlit pandas numpy xgboost plotly`
2. Guarda el código en un archivo llamado `app.py`.
3. Ejecuta en tu terminal: `streamlit run app.py`

```python
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from xgboost import XGBRegressor

# Configuración de la página Streamlit
st.set_page_config(
    page_title="Geometallurgy Plant Optimizer",
    page_icon="⛏️",
    layout="wide",
)


# ==========================================
# 1. ENTRENAMIENTO DEL MODELO BASE
# ==========================================
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 1000

    data = {
        "Kt": np.random.uniform(5, 35, n_samples),
        "CUT": np.random.uniform(0.3, 2.0, n_samples),
        "CUS": np.random.uniform(0.01, 0.3, n_samples),
        "CUCN": np.random.uniform(0.05, 1.8, n_samples),
        "FE": np.random.uniform(1.5, 4.0, n_samples),
        "PY": np.random.uniform(1.0, 5.0, n_samples),
        "TOTAR": np.random.uniform(0.5, 4.0, n_samples),
        "AxB": np.random.uniform(30, 130, n_samples),
    }
    df = pd.DataFrame(data)

    df["CuS_CuT_ratio"] = df["CUS"] / df["CUT"]
    df["CuCN_CuT_ratio"] = df["CUCN"] / df["CUT"]
    df["Fe_CuT_ratio"] = df["FE"] / df["CUT"]

    # Simulación de la respuesta metalúrgica
    df["REC"] = (
        78.0
        + (12.0 * df["CuCN_CuT_ratio"])
        - (3.5 * df["TOTAR"])
        - (1.2 * df["Fe_CuT_ratio"])
        + (0.05 * df["AxB"])
        - (2.0 * df["CuS_CuT_ratio"])
        + np.random.normal(0, 1.0, n_samples)
    )
    df["REC"] = np.clip(df["REC"], 60.0, 95.0)

    features = [
        "Kt",
        "CUT",
        "CUS",
        "CUCN",
        "FE",
        "PY",
        "TOTAR",
        "AxB",
        "CuS_CuT_ratio",
        "CuCN_CuT_ratio",
        "Fe_CuT_ratio",
    ]
    X = df[features]
    y = df["REC"]

    model = XGBRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42
    )
    model.fit(X, y)
    return model, features


model, feature_cols = train_model()

# ==========================================
# 2. INTERFAZ Y TITULO
# ==========================================
st.title("⛏️ Sistema Geometalúrgico de Predicción y Control Operacional")
st.markdown(
    "Optimización de setpoints de planta a partir de la información del plan de minado a corto plazo."
)

# Datos iniciales (Turno 13A por defecto)
default_data = pd.DataFrame({
    "PALA": ["SH007", "SH002", "SH002", "SH002"],
    "Poligono": [
        "F02N-3570-023_SSA1",
        "F03S-3630-020_SSB1",
        "F03S-3630-020_HYB1",
        "F03S-3630-020_SSA2",
    ],
    "Kt": [29.0, 16.0, 6.0, 22.0],
    "CUT": [1.86, 0.44, 0.35, 1.20],
    "CUS": [0.18, 0.04, 0.02, 0.14],
    "CUCN": [1.73, 0.19, 0.05, 1.13],
    "FE": [2.00, 3.12, 3.05, 3.21],
    "PY": [3.1, 3.4, 3.4, 3.3],
    "TOTAR": [2.5, 2.0, 1.2, 3.1],
    "AxB": [42.0, 66.0, 39.0, 129.0],
})

# ==========================================
# 3. EDITOR DE DATOS
# ==========================================
st.subheader("1. Plan de Alimentación a Planta (Tabla de Bloques)")
st.info(
    "Puedes editar los datos directamente en la tabla interactiva o agregar filas para simular el turno."
)

edited_df = st.data_editor(
    default_data, num_rows="dynamic", use_container_width=True
)

# ==========================================
# 4. PROCESAMIENTO DEL BLEND Y PREDICCIÓN
# ==========================================
if not edited_df.empty and edited_df["Kt"].sum() > 0:
    df_calc = edited_df.copy()

    # Ratios individuales por polígono
    df_calc["CuS_CuT_ratio"] = df_calc["CUS"] / df_calc["CUT"]
    df_calc["CuCN_CuT_ratio"] = df_calc["CUCN"] / df_calc["CUT"]
    df_calc["Fe_CuT_ratio"] = df_calc["FE"] / df_calc["CUT"]

    total_kt = df_calc["Kt"].sum()

    # Ponderación por tonelaje para la mezcla global
    blend_values = {}
    for col in [
        "Kt",
        "CUT",
        "CUS",
        "CUCN",
        "FE",
        "PY",
        "TOTAR",
        "AxB",
        "CuS_CuT_ratio",
        "CuCN_CuT_ratio",
        "Fe_CuT_ratio",
    ]:
        if col == "Kt":
            blend_values[col] = total_kt
        else:
            blend_values[col] = (
                df_calc[col] * df_calc["Kt"]
            ).sum() / total_kt

    df_blend = pd.DataFrame([blend_values])

    # Predicción con XGBoost
    pred_rec = model.predict(df_blend[feature_cols])[0]

    # ==========================================
    # 5. DASHBOARD DE RESULTADOS
    # ==========================================
    st.divider()
    st.subheader("2. Diagnóstico Geometalúrgico y Setpoints Operacionales")

    # Métricas principales
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Tonelaje Total", f"{total_kt:.1f} Kt")
    col2.metric("Recuperación Estimada", f"{pred_rec:.2f} %")
    col3.metric("Dureza Blend (A x b)", f"{df_blend['AxB'].iloc[0]:.1f}")
    col4.metric("Arcillas (TOTAR)", f"{df_blend['TOTAR'].iloc[0]:.2f}")
    col5.metric("Ratio Fe / CuT", f"{df_blend['Fe_CuT_ratio'].iloc[0]:.2f}")

    st.write("")

    # Prescripciones operativas
    axb_val = df_blend["AxB"].iloc[0]
    totar_val = df_blend["TOTAR"].iloc[0]
    fe_cut_val = df_blend["Fe_CuT_ratio"].iloc[0]

    c_molienda, c_reactivos, c_ph = st.columns(3)

    with c_molienda:
        st.markdown("### ⚙️ Molienda / SAG")
        if axb_val < 45:
            st.error(
                "**CRÍTICO:** Mineral duro.\n\n"
                "- Reducir TPH en 8-10%.\n"
                "- Incrementar carga de bolas.\n"
                "- Vigilancia en potencia del SAG."
            )
        elif axb_val > 90:
            st.success(
                "**ÓPTIMO:** Mineral blando.\n\n"
                "- Oportunidad de elevar TPH un +5%.\n"
                "- Monitorear sobremolienda de finos."
            )
        else:
            st.info(
                "**ESTÁNDAR:** Operación normal.\n\n"
                "- Mantener TPH objetivo del plan."
            )

    with c_reactivos:
        st.markdown("### 🧪 Espumante y Arcillas")
        if totar_val > 2.8:
            st.warning(
                "**ALERTA ARCILLAS ELEVADAS:**\n\n"
                "- Reducir dosificación de espumante -15%.\n"
                "- Monitorear viscosidad en Rougher.\n"
                "- Atento al arrastre de finos al concentrado."
            )
        else:
            st.info(
                "**ESTÁNDAR:** Contenido de arcillas normal.\n\n"
                "- Dosificación nominal según receta."
            )

    with c_ph:
        st.markdown("### ⚖️ Control de pH y Cal")
        if fe_cut_val > 5.0:
            st.warning(
                "**ALTO CONTENIDO DE PIRITA:**\n\n"
                "- Subir pH a 10.8 - 11.2 (mayor adición de cal).\n"
                "- Deprimir Fe para sostener el grado de concentrado."
            )
        else:
            st.info(
                "**CONTROL:** Relación Fe/CuT adecuada.\n\n"
                "- Mantener pH en rango 10.0 - 10.4."
            )

    # Gráfico de variabilidad por polígono
    st.divider()
    st.subheader("3. Análisis de Variabilidad entre Polígonos")

    fig = px.bar(
        edited_df,
        x="Poligono",
        y="Kt",
        color="AxB",
        hover_data=["CUT", "TOTAR", "PY"],
        title="Distribución de Tonelaje por Polígono coloreado por Dureza (A x b)",
        color_continuous_scale="Viridis_r",
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning(
        "Ingresa al menos un polígono con tonelaje mayor a 0 para generar las recomendaciones."
    )

```
