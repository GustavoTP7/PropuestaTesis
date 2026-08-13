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
import time

# --- CONFIGURACIÓN DE INTERFAZ ---
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
Integración de **Clustering Autónomo**, **Caracterización de Dominios** y **IA Explicable**.
""")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ 1. Arquitectura de Datos")
    archivo = st.file_uploader("Subir registros históricos", type=["csv", "xlsx"])
    modo_ruido = st.radio("Mitigación de Outliers [IQR]:", ["Data Original", "Depuración por IQR"])
    
    st.header("🤖 2. Motor de IA Autónomo")
    tipo_modelo = st.selectbox("Algoritmo:", ["XGBoost", "CatBoost"])
    balancear = st.checkbox("Balanceo SMOTE (Casos Críticos)")

    st.divider()
    ejecutar = st.button("🚀 Iniciar Simulación Digital", use_container_width=True)

if archivo is not None:
    df_raw = cargar_datos(archivo)
    
    if df_raw is not None:
        df_num = df_raw.select_dtypes(include=[np.number]).dropna()
        columnas = df_num.columns.tolist()
        
        with st.sidebar:
            st.header("🎯 3. Configuración de Variables")
            target = st.selectbox("Variable Respuesta (Y):", columnas, index=len(columnas)-1)
            features = st.multiselect("Variables Predictoras (X):", [c for c in columnas if c != target], 
                                     default=[c for c in columnas if c != target])

        if ejecutar:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # FASE 1: Limpieza
            status_text.text("Fase 1/5: Refinando datos...")
            df = df_num.copy()
            if modo_ruido == "Depuración por IQR":
                Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
            progress_bar.progress(20)

            # FASE 2: Dominios Geometalúrgicos Automáticos [5, 6]
            status_text.text("Fase 2/5: Identificando Unidades Geometalúrgicas (UGM)...")
            best_k, best_score = 2, -1
            for k in range(2, 6):
                if len(df) > k:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = km.fit_predict(df)
                    score = silhouette_score(df, labels)
                    if score > best_score:
                        best_score, best_k = score, k
            
            kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            df['Dominio_GMD'] = kmeans_final.fit_predict(df)
            
            # NUEVO: Generar resumen de los dominios para visualización [3, 4]
            df_resumen_dominios = df.groupby('Dominio_GMD')[features + [target]].mean()
            progress_bar.progress(40)

            # FASE 3: Balanceo
            X, y = df[features], df[target]
            if balancear:
                status_text.text("Fase 3/5: Aplicando SMOTE...")
                y_disc = pd.qcut(y, q=3, labels=False, duplicates='drop')
                sm = SMOTE(random_state=42, k_neighbors=min(2, len(X)-1))
                X, _ = sm.fit_resample(X, y_disc)
                y = df.loc[X.index, target]
            progress_bar.progress(60)

            # FASE 4: Entrenamiento
            status_text.text(f"Fase 4/5: Entrenando {tipo_modelo}...")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            if tipo_modelo == "XGBoost":
                model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, random_state=42, verbose=0, early_stopping_rounds=50)
                model.fit(X_train, y_train, eval_set=(X_val, y_val))
            progress_bar.progress(80)

            # FASE 5: Validación
            status_text.text("Fase 5/5: Validando fidelidad predictiva...")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            y_pred = cross_val_predict(model, X, y, cv=kf)
            r2, mae, rmse = r2_score(y, y_pred), mean_absolute_error(y, y_pred), np.sqrt(mean_squared_error(y, y_pred))
            mape = mean_absolute_percentage_error(y, y_pred) * 100
            progress_bar.progress(100)
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Calidad de Datos", "🎯 Score de Precisión", "🎛️ Simulador", "🚨 Monitor FDI", "🧠 Transparencia IA"
            ])

            with tab1:
                st.subheader("Análisis Multivariante y Caracterización de Dominios")
                
                # NUEVO: Tabla de Información de Dominios [4, 7]
                st.write(f"### 📍 Perfil Técnico de las {best_k} Unidades Geometalúrgicas (UGM)")
                st.markdown("""
                Esta tabla muestra los valores promedio por dominio. Úsela para identificar qué grupo 
                representa mineral complejo (ej. mayor Factor K) o de alta ley [2].
                """)
                st.dataframe(df_resumen_dominios.style.background_gradient(cmap='viridis'))
                
                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    var_sel = st.selectbox("Analizar tendencia vs " + target, features)
                    st.plotly_chart(px.scatter(df, x=var_sel, y=target, color='Dominio_GMD', trendline="ols", 
                                             title=f"Dispersión por Dominios: {var_sel}"), use_container_width=True)
                with c2:
                    st.plotly_chart(px.imshow(df[[target] + features].corr(), text_auto=".2f", 
                                             color_continuous_scale="RdBu_r", title="Mapa de Interdependencia"), use_container_width=True)

            with tab2:
                st.subheader("Evaluación de la Fidelidad Predictiva")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Fidelidad (R²)", f"{r2:.3f}")
                m2.metric("Error (MAE)", f"{mae:.3f}")
                m3.metric("Riesgo (RMSE)", f"{rmse:.3f}")
                m4.metric("MAPE", f"{mape:.2f}%")
                
                c_rp, c_im = st.columns(2)
                with c_rp:
                    st.plotly_chart(px.scatter(x=y, y=y_pred, labels={'x': 'Realidad', 'y': 'Predicción'}, title="Realidad vs Digital Twin"), use_container_width=True)
                with c_im:
                    importances = model.feature_importances_ if tipo_modelo == "XGBoost" else model.get_feature_importance()
                    df_imp = pd.DataFrame({'Variable': features, 'Importancia': importances}).sort_values('Importancia', ascending=True)
                    st.plotly_chart(px.bar(df_imp, x='Importancia', y='Variable', orientation='h', title="Atribución de Relevancia"), use_container_width=True)

            with tab3:
                st.subheader("Simulación Prescriptiva: Análisis 'What-If'")
                inputs_sim = {col: st.slider(f"{col}:", float(df[col].min()), float(df[col].max()), float(df[col].mean())) for col in features}
                df_sim_input = pd.DataFrame([inputs_sim])
                pred_raw = model.predict(df_sim_input)
                pred_sim = pred_raw.item() if hasattr(pred_raw, "item") else pred_raw
                
                if st.button("🚀 Maximizar Recuperación"):
                    best_v, best_cfg = -1, None
                    for _ in range(200):
                        rand_cfg = {c: np.random.uniform(df[c].min(), df[c].max()) for c in features}
                        p_rand = model.predict(pd.DataFrame([rand_cfg]))
                        val_p = p_rand.item() if hasattr(p_rand, "item") else p_rand
                        if val_p > best_v: best_v, best_cfg = val_p, rand_cfg
                    st.success(f"Máximo Potencial Identificado: {best_v:.2f}%")
                    st.json(best_cfg)
                st.metric(label=f"{target} Estimado", value=f"{pred_sim:.2f}%")

            with tab4:
                st.subheader("Detección e Isolation de Fallas (FDI)")
                df_audit = df.copy()
                df_audit['Error'] = np.abs(df[target] - y_pred[:len(df)])
                def semaforo(e):
                    if e <= mae: return "🟢 Normal"
                    elif e <= 2*mae: return "🟡 Advertencia"
                    else: return "🔴 Anomalía"
                df_audit['Estado'] = df_audit['Error'].apply(semaforo)
                st.plotly_chart(px.scatter(df_audit, x=df_audit.index, y='Error', color='Estado', 
                                         color_discrete_map={"🟢 Normal": "green", "🟡 Advertencia": "orange", "🔴 Anomalía": "red"},
                                         title="Protocolo de Auditoría de Turnos (FDI)"), use_container_width=True)
                st.write("**Resumen de Alertas Operativas por Turno:**")
                st.dataframe(df_audit['Estado'].value_counts())

            with tab5:
                st.subheader("IA Explicable (XAI)")
                X_shap = X.sample(min(100, len(X)))
                explainer = shap.Explainer(model, X_shap)
                shap_v = explainer(X_shap)
                fig_s, _ = plt.subplots()
                shap.summary_plot(shap_v, X_shap, show=False)
                st.pyplot(fig_s)
        else:
            st.info("💡 Configure los parámetros y presione el botón para ver la caracterización de dominios y semáforos.")
else:
    st.info("👈 Cargue el dataset histórico para iniciar el Digital Twin.")
