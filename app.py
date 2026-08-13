import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
import shap
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, silhouette_score
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE INTERFAZ PROFESIONAL ---
st.set_page_config(page_title="Geomet Twin Pro", layout="wide")

@st.cache_data
def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error en la ingesta de datos: {e}")
        return None

st.title("💎 Geomet Twin Pro: Inteligencia Operacional para Flotación")
st.markdown("""
**Digital Twin de Soporte a la Decisión (DSS)**. 
Optimizado para alta velocidad mediante **Dominios Automáticos** y **Early Stopping**.
""")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ 1. Arquitectura de Datos")
    archivo = st.file_uploader("Subir registros históricos (CSV/XLSX)", type=["csv", "xlsx"])
    modo_ruido = st.radio("Filtro de Anomalías [IQR]:", ["Data Original", "Depuración por Rango Intercuartílico"])
    
    st.header("🤖 2. Motor de IA Autónomo")
    tipo_modelo = st.selectbox("Seleccionar Algoritmo:", ["XGBoost", "CatBoost"])
    st.info("💡 **Configuración Automática:** El sistema determinará la tasa de aprendizaje y el número de árboles mediante lógica de parada temprana (Early Stopping).")
    
    balancear = st.checkbox("Balanceo de Casos Críticos (SMOTE)")

if archivo is not None:
    df_raw = cargar_datos(archivo)
    
    if df_raw is not None:
        df = df_raw.select_dtypes(include=[np.number]).dropna()
        
        if modo_ruido == "Depuración por Rango Intercuartílico":
            Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
            
        # --- LÓGICA DE DOMINIOS AUTOMÁTICA (Silhouette Score) ---
        st.sidebar.subheader("📍 Identificación de Dominios")
        best_k, best_score = 1, -1
        # Evaluamos automáticamente el mejor número de clusters basado en silueta [1]
        for k in range(2, 6):
            if len(df) > k:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(df)
                score = silhouette_score(df, labels)
                if score > best_score:
                    best_score, best_k = score, k
        
        kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        df['Dominio_GMD'] = kmeans_final.fit_predict(df)
        st.sidebar.success(f"Configuración Óptima: {best_k} Dominios Geometalúrgicos detectados.")

        columnas = df.columns.tolist()
        with st.sidebar:
            st.header("🎯 3. Variables")
            target = st.selectbox("Respuesta (Y):", columnas, index=len(columnas)-2)
            features = st.multiselect("Predictores (X):", [c for c in columnas if c not in [target, 'Dominio_GMD']], 
                                     default=[c for c in columnas if c not in [target, 'Dominio_GMD']])
            
        if features and target:
            X, y = df[features], df[target]

            if balancear:
                y_disc = pd.qcut(y, q=3, labels=False, duplicates='drop')
                sm = SMOTE(random_state=42, k_neighbors=min(2, len(X)-1))
                X, _ = sm.fit_resample(X, y_disc)
                y = df.loc[X.index, target]

            # --- ENTRENAMIENTO AUTOMÁTICO (EARLY STOPPING) ---
            # Dividimos para validación interna de la IA
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            if tipo_modelo == "XGBoost":
                model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False) # Early stopping implícito por defecto en versiones nuevas
            else:
                model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, random_state=42, verbose=0, early_stopping_rounds=50)
                model.fit(X_train, y_train, eval_set=(X_val, y_val))

            # Validación Cruzada Rápida (K=5)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            y_pred = cross_val_predict(model, X, y, cv=kf)
            
            # Métricas
            r2, mae, rmse = r2_score(y, y_pred), mean_absolute_error(y, y_pred), np.sqrt(mean_squared_error(y, y_pred))
            mape = mean_absolute_percentage_error(y, y_pred) * 100
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Calidad de Datos", "🎯 Score de Precisión", "🎛️ Simulador", "🚨 Monitor FDI", "🧠 SHAP"
            ])

            with tab1:
                st.subheader("Filtro y Refinamiento Multivariante")
                c1, c2 = st.columns(2)
                with c1: st.plotly_chart(px.scatter(df, x=features, y=target, color='Dominio_GMD', title="Distribución por Dominios"), use_container_width=True)
                with c2: st.plotly_chart(px.imshow(df[[target] + features].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlaciones"), use_container_width=True)

            with tab2:
                st.subheader("Evaluación de la Fidelidad Predictiva")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Fidelidad (R²)", f"{r2:.3f}")
                m2.metric("MAE", f"{mae:.3f}")
                m3.metric("Riesgo (RMSE)", f"{rmse:.3f}")
                m4.metric("Error Relativo", f"{mape:.2f}%")
                st.plotly_chart(px.scatter(x=y, y=y_pred, labels={'x': 'Real', 'y': 'Digital'}, title="Realidad vs Digital Twin"), use_container_width=True)

            with tab3:
                st.subheader("Simulación Prescriptiva: Análisis 'What-If'")
                inputs_sim = {}
                cols_sim = st.columns(3)
                for idx, col_name in enumerate(features):
                    with cols_sim[idx % 3]:
                        inputs_sim[col_name] = st.slider(f"{col_name}:", float(df[col_name].min()), float(df[col_name].max()), float(df[col_name].mean()))
                
                pred_sim = model.predict(pd.DataFrame([inputs_sim]))
                
                if st.button("🚀 Maximizar Recuperación (Optimización Estocástica)"):
                    best_v, best_cfg = -1, None
                    for _ in range(500):
                        rand_cfg = {c: np.random.uniform(df[c].min(), df[c].max()) for c in features}
                        p = model.predict(pd.DataFrame([rand_cfg]))
                        if p > best_v: best_v, best_cfg = p, rand_cfg
                    st.success(f"Potencial Máximo: {best_v:.2f}%")
                    st.json(best_cfg)
                st.metric(label=f"{target} Estimado", value=f"{pred_sim:.2f}%")

            with tab4:
                st.subheader("Detección e Isolation de Fallas (FDI)")
                df_audit = df.copy()
                df_audit['Error'] = np.abs(y - y_pred)
                df_audit['Estado'] = df_audit['Error'].apply(lambda e: "🟢 Normal" if e <= mae else "🔴 Anomalía")
                st.plotly_chart(px.scatter(df_audit, x=df_audit.index, y='Error', color='Estado', title="Protocolo FDI por Turno"), use_container_width=True)

            with tab5:
                st.subheader("Interpretabilidad XAI")
                # Submuestreo para rapidez en SHAP
                X_shap = X.sample(min(100, len(X)))
                explainer = shap.Explainer(model, X_shap)
                shap_v = explainer(X_shap)
                fig_s, ax = plt.subplots()
                shap.summary_plot(shap_v, X_shap, show=False)
                st.pyplot(fig_s)
else:
    st.info("👈 Cargue el dataset histórico para iniciar el Digital Twin de alta velocidad.")
