import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
import shap
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import optuna 
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

# --- TÍTULO COMERCIAL ---
st.title("💎 Geomet Twin Pro: Inteligencia Operacional para Flotación")
st.markdown("""
**Sistema Inteligente de Soporte a la Decisión (DSS)** basado en registros históricos operacionales. 
Este Digital Twin integra **Optimización Bayesiana**, **Dominios Geometalúrgicos** y **IA Explicable**.
""")

# --- BARRA LATERAL: ARQUITECTURA DE DATOS Y MODELO ---
with st.sidebar:
    st.header("⚙️ 1. Ingesta y Refinamiento de Datos")
    archivo = st.file_uploader("Subir registros históricos (CSV/XLSX)", type=["csv", "xlsx"])
    
    modo_ruido = st.radio("Mitigación de Outliers [IQR]:", ["Data Original", "Depuración por Rango Intercuartílico"])
    
    # Mejora: Definición de Unidades Geometalúrgicas (UGM) [3, 4]
    n_clusters = st.slider("Identificación de Dominios (Clusters):", 1, 5, 1)

    st.header("🤖 2. Configuración del Cerebro IA")
    tipo_modelo = st.selectbox("Algoritmo de Aprendizaje:", ["XGBoost", "CatBoost"])
    
    # Mejora: Optimización Heurística de Hiperparámetros [5, 6]
    metodo_tuning = st.radio("Ajuste de Parámetros:", ["Manual", "Auto-Tuning (TPE/Optuna)"])
    
    if metodo_tuning == "Manual":
        n_estimators = st.slider("Iteraciones de Boosting (Árboles):", 50, 1000, 100)
        lr = st.slider("Tasa de Aprendizaje (Learning Rate):", 0.01, 0.3, 0.05)
    else:
        n_trials = st.number_input("Ciclos de Optimización Bayesiana:", 10, 50, 20)
    
    # Mejora: Balanceo de muestra mediante SMOTE [5, 7]
    balancear = st.checkbox("Balanceo Sintético de Casos Críticos (SMOTE)")

if archivo is not None:
    df_raw = cargar_datos(archivo)
    
    if df_raw is not None:
        # Selección de columnas numéricas y eliminación de nulos [8]
        df = df_raw.select_dtypes(include=[np.number]).dropna()
        
        # Filtro de Outliers por IQR [8]
        if modo_ruido == "Depuración por Rango Intercuartílico":
            Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
            
        # Lógica de Dominios Geometalúrgicos (K-Means) [3]
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['Dominio_GMD'] = kmeans.fit_predict(df)
            
        columnas = df.columns.tolist()
        
        with st.sidebar:
            st.header("🎯 3. Definición de Variables")
            target = st.selectbox("Variable Objetivo (Respuesta):", columnas, index=len(columnas)-1)
            features = st.multiselect("Variables Manipuladas y Perturbaciones:", [c for c in columnas if c != target], 
                                     default=[c for c in columnas if c != target])
            
        if features and target:
            X, y = df[features], df[target]

            # Lógica SMOTE para balanceo de casos de baja recuperación [5]
            if balancear:
                y_disc = pd.qcut(y, q=3, labels=False, duplicates='drop')
                sm = SMOTE(random_state=42, k_neighbors=min(2, len(X)-1))
                X, _ = sm.fit_resample(X, y_disc)
                y = df.loc[X.index, target]

            # --- MOTOR DE OPTIMIZACIÓN BAYESIANA (OPTUNA/TPE) [9] ---
            if metodo_tuning == "Auto-Tuning (TPE/Optuna)":
                def objective(trial):
                    p = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'max_depth': trial.suggest_int('max_depth', 3, 10)
                    }
                    m = xgb.XGBRegressor(**p, random_state=42) if tipo_modelo == "XGBoost" else CatBoostRegressor(**p, verbose=0)
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    y_p = cross_val_predict(m, X, y, cv=kf)
                    return np.sqrt(mean_squared_error(y, y_p))

                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=n_trials)
                best_params = study.best_params
            else:
                best_params = {'n_estimators': n_estimators, 'learning_rate': lr}

            # Entrenamiento Final con K-Fold [10, 11]
            model = xgb.XGBRegressor(**best_params, random_state=42) if tipo_modelo == "XGBoost" else CatBoostRegressor(**best_params, verbose=0)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            y_pred = cross_val_predict(model, X, y, cv=kf)
            model.fit(X, y)

            # Métricas de Fidelidad Predictiva [11, 12]
            r2, mae, rmse = r2_score(y, y_pred), mean_absolute_error(y, y_pred), np.sqrt(mean_squared_error(y, y_pred))
            mape = mean_absolute_percentage_error(y, y_pred) * 100
            
            # --- TABS CON TÍTULOS COMERCIALES ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Calidad de Datos", 
                "🎯 Score de Precisión", 
                "🎛️ Simulador de Escenarios", 
                "🚨 Monitor de Desviaciones",
                "🧠 Transparencia IA (SHAP)"
            ])

            with tab1:
                st.subheader("Filtro y Refinamiento de registros históricos")
                c1, c2 = st.columns(2)
                with c1:
                    var_v = st.selectbox("Correlación Bivariante vs " + target, features)
                    st.plotly_chart(px.scatter(df, x=var_v, y=target, trendline="ols", title=f"Tendencia: {var_v}"), use_container_width=True)
                with c2:
                    st.plotly_chart(px.imshow(df[[target] + features].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", title="Mapa de Correlación Multivariante"), use_container_width=True)

            with tab2:
                st.subheader("Evaluación de la Fidelidad Predictiva")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Confiabilidad (R²)", f"{r2:.3f}")
                m2.metric("Error Absoluto (MAE)", f"{mae:.3f}")
                m3.metric("Riesgo (RMSE)", f"{rmse:.3f}")
                m4.metric("Error Relativo (MAPE)", f"{mape:.2f}%")
                
                c_rp, c_im = st.columns(2)
                with c_rp:
                    fig_rp = px.scatter(x=y, y=y_pred, labels={'x': 'Realidad Operativa', 'y': 'Predicción Digital'}, title="Fidelidad Real vs Digital")
                    fig_rp.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="Red", dash="dash"))
                    st.plotly_chart(fig_rp, use_container_width=True)
                with c_im:
                    importances = model.feature_importances_ if tipo_modelo == "XGBoost" else model.get_feature_importance()
                    df_imp = pd.DataFrame({'Atributo': features, 'Impacto': importances}).sort_values('Impacto', ascending=True)
                    st.plotly_chart(px.bar(df_imp, x='Impacto', y='Atributo', orientation='h', title="Ranking de Relevancia de Atributos"), use_container_width=True)

            with tab3:
                st.subheader("Simulación Prescriptiva: Análisis 'What-If'")
                st.markdown("Ajuste los **Set-Points** operativos para previsualizar la respuesta metalúrgica:")
                
                inputs_sim = {}
                cols_sim = st.columns(3)
                for idx, col_name in enumerate(features):
                    with cols_sim[idx % 3]:
                        inputs_sim[col_name] = st.slider(f"{col_name}:", float(df[col_name].min()), float(df[col_name].max()), float(df[col_name].mean()))
                        
                df_sim_input = pd.DataFrame([inputs_sim])
                pred_sim = model.predict(df_sim_input)
                
                st.divider()
                # MEJORA: OPTIMIZACIÓN PROACTIVA DE SET-POINTS [13, 14]
                if st.button("🚀 Calcular Configuración de Máxima Recuperación"):
                    best_val, best_cfg = -1, None
                    for _ in range(300):
                        rand_cfg = {c: np.random.uniform(df[c].min(), df[c].max()) for c in features}
                        p_rand = model.predict(pd.DataFrame([rand_cfg]))
                        if p_rand > best_val: best_val, best_cfg = p_rand, rand_cfg
                    st.success(f"Potencial Máximo Identificado: {best_val:.2f}%")
                    st.json(best_cfg)

                st.metric(label=f"{target} Estimado", value=f"{pred_sim:.2f}%")

            with tab4:
                st.subheader("Detección de Anomalías y Auditoría de Turnos [FDI]")
                df_audit = df.copy()
                df_audit['Error'] = np.abs(y - y_pred)
                def cat_error(e): return "🟢 Normal" if e <= mae else "🟡 Advertencia" if e <= 2*mae else "🔴 Anomalía"
                df_audit['Estado'] = df_audit['Error'].apply(cat_error)
                st.plotly_chart(px.scatter(df_audit, x=df_audit.index, y='Error', color='Estado', title="Protocolo de Detección e Isolation de Fallas (Fault Detection)"), use_container_width=True)

            with tab5:
                st.subheader("Interpretabilidad mediante IA Explicable (XAI)")
                st.markdown("Cuantificación del impacto de cada variable mediante **Valores de Shapley** [15, 16]:")
                explainer = shap.Explainer(model, X)
                shap_v = explainer(X)
                fig_s, ax = plt.subplots()
                shap.summary_plot(shap_v, X, show=False)
                st.pyplot(fig_s)
else:
    st.info("👈 Por favor, cargue el dataset histórico en el panel lateral para iniciar el Digital Twin.")
