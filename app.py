import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Sistema Geometalúrgico MineStar", page_icon="⚒️", layout="wide"
)

st.title("⚒️ Gemelo Digital Geometalúrgico - Integración MineStar")
st.markdown(
    "Procesamiento directo de reportes operacionales de **MineStar**, modelos de bloques y control de planta."
)

# ----------------------------------------------------
# 1. CARGA Y LIMPIEZA NATIVA DE MINESTAR
# ----------------------------------------------------
st.sidebar.header("📁 Carga de Datos MineStar")
uploaded_file = st.sidebar.file_uploader(
    "Subir archivo MineStar (.xlsx / .csv)", type=["xlsx", "csv"]
)


def load_and_clean_minestar(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # Filtrar filas de totales o notas al pie generadas por MineStar
    if "Shift" in df.columns:
        df = df[df["Shift"].notna()]
        df = df[~df["Shift"].astype(str).str.startswith("Total")]
        df = df[~df["Shift"].astype(str).str.startswith("Applied filters")]

    if "Mining Block" in df.columns:
        df = df[df["Mining Block"].notna()]

    # Convertir columnas numéricas de forma segura
    numeric_cols = [
        "Payload",
        "CuT",
        "CuS",
        "CuCN",
        "MoT",
        "Fe",
        "S",
        "SagTph",
        "pltTph",
        "bmTph",
        "RecCu",
        "RecMo",
        "CPY",
        "CC",
        "CV",
        "BN",
        "PY",
        "Cao_18",
        "Mont_18",
        "Filo_18",
        "BWI",
        "AXB23",
        "UCS",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


if uploaded_file is not None:
    df_raw = load_and_clean_minestar(uploaded_file)
    st.sidebar.success(f"Cargados {len(df_raw)} registros de MineStar")
else:
    st.info(
        "💡 Por favor sube tu archivo `MineStar 30-10-2025.xlsx` para activar el panel."
    )
    st.stop()

# ----------------------------------------------------
# 2. FEATURE ENGINEERING GEOMETALÚRGICO
# ----------------------------------------------------
df = df_raw.copy()

# Ratios de Cobre Soluble
if "CuT" in df.columns and "CuS" in df.columns:
    df["Ratio_CuS_CuT"] = df["CuS"] / np.where(df["CuT"] == 0, 0.001, df["CuT"])

if "CuT" in df.columns and "CuCN" in df.columns:
    df["Ratio_CuCN_CuT"] = df["CuCN"] / np.where(
        df["CuT"] == 0, 0.001, df["CuT"]
    )

# Ratios Mineralógicos (Sulfuros Secundarios vs Primarios)
if all(col in df.columns for col in ["CC", "CV", "CPY", "BN"]):
    sec_cu = df["CC"].fillna(0) + df["CV"].fillna(0)
    prim_cu = df["CPY"].fillna(0) + df["BN"].fillna(0)
    df["Ratio_Sec_Prim_Cu"] = sec_cu / np.where(prim_cu == 0, 0.001, prim_cu)

# Total Arcillas
clay_cols = [c for c in ["Cao_18", "Mont_18", "Filo_18"] if c in df.columns]
if clay_cols:
    df["Total_Clays"] = df[clay_cols].sum(axis=1)

# Ratio Pirita / Cobre
if "PY" in df.columns and "CuT" in df.columns:
    df["Ratio_PY_CuT"] = df["PY"] / np.where(df["CuT"] == 0, 0.001, df["CuT"])

# ----------------------------------------------------
# 3. INTERFAZ INTERACTIVA Y MODELADO
# ----------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📈 1. Resumen de Turno & Balance",
    "🤖 2. Modelo Predictivo (Recuperación / TPH)",
    "🎛️ 3. Sensibilidad Mineralógica",
])

with tab1:
    st.subheader("📊 Métricas Consolidadas del Reporte MineStar")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Tonelaje Total (t)",
        f"{df['Payload'].sum():,.1f}" if "Payload" in df.columns else "N/A",
    )
    m2.metric(
        "Ley Promedio %CuT",
        f"{df['CuT'].mean():.3f}%" if "CuT" in df.columns else "N/A",
    )
    m3.metric(
        "Tratamiento Promedio (TPH)",
        f"{df['pltTph'].mean():,.1f}" if "pltTph" in df.columns else "N/A",
    )
    m4.metric(
        "Recuperación Promedio %RecCu",
        f"{df['RecCu'].mean():.2f}%" if "RecCu" in df.columns else "N/A",
    )

    st.markdown("### Comportamiento por Bloque de Minado")
    fig_block = px.bar(
        df,
        x="Mining Block",
        y="Payload",
        color="CuT" if "CuT" in df.columns else None,
        hover_data=["pltTph", "RecCu"] if "pltTph" in df.columns else [],
        title="Tonelaje por Bloque de Minado y Ley de Cobre Total (%CuT)",
        labels={"Payload": "Tonelaje (t)", "CuT": "%CuT"},
    )
    st.plotly_chart(fig_block, use_container_width=True)

with tab2:
    st.subheader("🤖 Entrenamiento de Modelo con Datos de MineStar")

    target_var = st.selectbox(
        "Seleccionar Variable Objetivo a Predecir",
        ["RecCu", "pltTph", "SagTph"],
        index=0,
    )

    # Identificar predictors numéricos
    features = [
        c
        for c in [
            "CuT",
            "CuS",
            "CuCN",
            "MoT",
            "Fe",
            "S",
            "CPY",
            "CC",
            "CV",
            "BN",
            "PY",
            "Total_Clays",
            "Ratio_CuS_CuT",
            "Ratio_Sec_Prim_Cu",
            "BWI",
            "AXB23",
            "UCS",
        ]
        if c in df.columns
    ]

    valid_df = df.dropna(subset=[target_var] + features)

    if len(valid_df) >= 5:
        X = valid_df[features]
        y = valid_df[target_var]

        model = GradientBoostingRegressor(
            n_estimators=50, random_state=42, max_depth=3
        )
        model.fit(X, y)

        imp_df = (
            pd.DataFrame(
                {"Variable": features, "Importancia": model.feature_importances_}
            )
            .sort_values("Importancia", ascending=True)
        )

        fig_imp = px.bar(
            imp_df,
            x="Importancia",
            y="Variable",
            orientation="h",
            title=f"Importancia de Variables para {target_var}",
            color="Importancia",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning(
            "Se requieren más registros completos en la muestra para entrenar el modelo predictivo."
        )

with tab3:
    st.subheader("🔍 Impacto de Arcillas y Mineralogía en el Tratamiento")

    if "Total_Clays" in df.columns and "pltTph" in df.columns:
        fig_scatter = px.scatter(
            df,
            x="Total_Clays",
            y="pltTph",
            color="RecCu" if "RecCu" in df.columns else None,
            size="Payload" if "Payload" in df.columns else None,
            hover_name="Mining Block",
            title="Efecto del Contenido de Arcillas (%Filosilicatos + Caolín + Montmorillonita) sobre el TPH",
            labels={
                "Total_Clays": "Arcillas Totales (%)",
                "pltTph": "Tratamiento Planta (TPH)",
            },
            trendline="ols",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Tabla Transaccional Procesada")
    st.dataframe(df, use_container_width=True)
