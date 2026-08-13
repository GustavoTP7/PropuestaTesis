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
        df = df.loc[:, ~df.columns.duplicated()] 
        return df
    except Exception as e:
        st.error(f"Error en la ingesta de datos: {e}")
        return None

st.title("💎 Geomet Twin Pro: Inteligencia Operacional para Flotación")
st.markdown("""
**Digital Twin de Soporte a la Decisión (DSS)**. 
Exploración de datos robusta, caracterización de dominios y auditoría técnica avanzada.
""")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ 1. Arquitectura de Datos")
    archivo = st.file_uploader("Subir registros históricos", type=["csv", "xlsx"])
    modo_ruido = st.radio("Filtro de Outliers [IQR]:", ["Data Original", "Depuración por IQR"])
    
    st.header("🤖 2. Motor de IA Autónomo")
    tipo_modelo = st.selectbox("Seleccionar Algoritmo:", ["XGBoost", "CatBoost"])
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
            target = st.selectbox("Variable Objetivo (Respuesta):", columnas, index=len(columnas)-1)
            features = st.multiselect("Predictores (X):", [c for c in columnas if c != target], 
                                     default=[c for c in columnas if c != target])

        if ejecutar or 'model' in st.session_state:
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

                # FASE 2: Dominios Automáticos
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
                progress_bar.progress(40)

                # FASE 3: SMOTE (CORREGIDO PARA EVITAR KEYERROR)
                X, y = df[features], df[target]
                if balancear:
                    status_text.text("Fase 3/5: Aplicando SMOTE...")
                    # Discretización para definir categorías a balancear [7]
                    y_disc = pd.qcut(y, q=3, labels=False, duplicates='drop')
                    sm = SMOTE(random_state=42, k_neighbors=min(2, len(X)-1))
                    
                    # Técnica para balancear X e Y (continuo) al mismo tiempo
                    X_with_y = X.copy()
                    X_with_y['__target_temp__'] = y
                    X_resampled, _ = sm.fit_resample(X_with_y, y_disc)
                    
                    y = X_resampled['__target_temp__']
                    X = X_resampled.drop(columns=['__target_temp__'])
                progress_bar.progress(60)

                # FASE 4: Entrenamiento IA
                status_text.text(f"Fase 4/5: Entrenando cerebro {tipo_modelo}...")
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                if tipo_modelo == "XGBoost":
                    model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, random_state=42, verbose=0, early_stopping_rounds=50)
                    model.fit(X_train, y_train, eval_set=(X_val, y_val))
                progress_bar.progress(80)

                # FASE 5: Validación Cruzada
                status_text.text("Fase 5/5: Validando Digital Twin...")
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                y_pred_cv = cross_val_predict(model, X, y, cv=kf)
                
                # PERSISTENCIA
                st.session_state.model = model
                st.session_state.df_proc = df # DataFrame con Dominios
                st.session_state.y_pred = y_pred_cv
                st.session_state.metrics = (r2_score(y, y_pred_cv), mean_absolute_error(y, y_pred_cv), np.sqrt(mean_squared_error(y, y_pred_cv)), mean_absolute_percentage_error(y, y_pred_cv) * 100)
                st.session_state.best_k = best_k
                st.session_state.X_final = X
                st.session_state.y_final = y

                progress_bar.progress(100)
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

            # Recuperar datos
            model, df_p = st.session_state.model, st.session_state.df_proc
            y_pred, y_final = st.session_state.y_pred, st.session_state.y_final
            r2, mae, rmse, mape = st.session_state.metrics
            best_k, X_final = st.session_state.best_k, st.session_state.X_final

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📈 Calidad de Datos", "📊 Análisis Multivariante", "🎯 Score de Precisión", "🎛️ Simulador", "🚨 Monitor FDI", "🧠 XAI"
            ])

            with tab1:
                st.subheader("Análisis Exploratorio y Caracterización de Dominios")
                df_resumen = df_p.groupby('Dominio_GMD')[features + [target]].mean()
                st.write(f"### 📍 Perfil Técnico de las {best_k} Unidades Geometalúrgicas (UGM)")
                st.dataframe(df_resumen.style.background_gradient(cmap='viridis'))
                
                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    var_x = st.selectbox("Variable Eje X:", df_p.columns, index=0, key="sel_x")
                    var_y = st.selectbox("Variable Eje Y:", df_p.columns, index=df_p.columns.tolist().index(target), key="sel_y")
                    actualizar_grafico = st.button("🔄 Actualizar Visualización")
                with c2:
                    if 'fig_explorador' not in st.session_state or actualizar_grafico:
                        if var_x == var_y:
                            st.session_state.fig_explorador = px.histogram(df_p, x=var_x, color='Dominio_GMD', title=f"Distribución de {var_x}")
                        else:
                            st.session_state.fig_explorador = px.scatter(df_p, x=var_x, y=var_y, color='Dominio_GMD', trendline="ols", title=f"Relación: {var_x} vs {var_y}")
                    st.plotly_chart(st.session_state.fig_explorador, use_container_width=True)

            with tab2:
                st.subheader("Interdependencia y Atribución Tecnológica")
                c_heat, c_imp = st.columns(2)
                with c_heat:
                    st.write("**Mapa de Interdependencia (Heatmap)**")
                    corr_matrix = df_p[[target] + features].corr()
                    st.plotly_chart(px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r"), use_container_width=True)
                with c_imp:
                    st.write("**Ranking de Importancia de Variables**")
                    importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.get_feature_importance()
                    df_imp = pd.DataFrame({'Variable': features, 'Impacto': importances}).sort_values('Impacto', ascending=True)
                    st.plotly_chart(px.bar(df_imp, x='Impacto', y='Variable', orientation='h', title="Peso de Atributos"), use_container_width=True)

            with tab3:
                st.subheader("Evaluación de la Fidelidad Predictiva")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Fidelidad (R²)", f"{r2:.3f}")
                m2.metric("Error (MAE)", f"{mae:.3f}")
                m3.metric("Riesgo (RMSE)", f"{rmse:.3f}")
                m4.metric("Error Relativo", f"{mape:.2f}%")
                st.plotly_chart(px.scatter(x=y_final, y=y_pred, labels={'x': 'Realidad Operativa', 'y': 'Digital Twin'}, 
                                         trendline="ols", title="Fidelidad Real vs Digital"), use_container_width=True)

            with tab4:
                st.subheader("Simulación Prescriptiva: Análisis 'What-If'")
                inputs_sim = {col: st.slider(f"{col}:", float(df_p[col].min()), float(df_p[col].max()), float(df_p[col].mean()), key=f"sim_{col}") for col in features}
                df_sim_input = pd.DataFrame([inputs_sim])
                pred_raw = model.predict(df_sim_input)
                pred_sim = pred_raw.item() if hasattr(pred_raw, "item") else pred_raw
                st.metric(label=f"{target} Estimado", value=f"{pred_sim:.2f}%")

            with tab5:
                st.subheader("Protocolo FDI: Auditoría de Registros")
                # Creamos un dataframe de auditoría basado en los datos balanceados finales
                df_audit = X_final.copy()
                df_audit[target] = y_final
                df_audit['Predicción'] = y_pred
                df_audit['Error'] = np.abs(df_audit[target] - df_audit['Predicción'])
                
                def logic_semaforo(e):
                    if e <= mae: return "🟢 Normal"
                    elif e <= 2*mae: return "🟡 Advertencia"
                    else: return "🔴 Anomalía"
                
                df_audit['Alerta FDI'] = df_audit['Error'].apply(logic_semaforo)
                
                st.write("**Tabla de registros (incluye casos balanceados) vs Digital Twin:**")
                st.dataframe(df_audit[[target, 'Predicción', 'Error', 'Alerta FDI'] + features].head(1000).style.map(
                    lambda x: "background-color: #90EE90" if x == "🟢 Normal" else ("background-color: #FFD700" if x == "🟡 Advertencia" else ("background-color: #F08080" if x == "🔴 Anomalía" else "")),
                    subset=['Alerta FDI']
                ))

            with tab6:
                st.subheader("IA Explicable (Transparencia)")
                X_shap = X_final.sample(min(100, len(X_final)))
                explainer = shap.Explainer(model, X_shap)
                shap_v = explainer(X_shap)
                fig_s, _ = plt.subplots()
                shap.summary_plot(shap_v, X_shap, show=False)
                st.pyplot(fig_s)
        else:
            st.info("💡 Configure los parámetros y pulse 'Iniciar Simulación Digital' para procesar datos.")
else:
    st.info("👈 Cargue el dataset histórico para iniciar el Digital Twin.")
