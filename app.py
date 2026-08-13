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

st.title("💎 Geomet Twin Pro: Inteligencia Operacional")
st.markdown("**Digital Twin de Soporte a la Decisión (DSS)** para optimización de procesos mineros.")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ 1. Arquitectura de Datos")
    archivo = st.file_uploader("Subir registros históricos", type=["csv", "xlsx"])
    modo_ruido = st.radio("Filtro de Outliers [IQR]:", ["Data Original", "Depuración por IQR"])
    
    st.header("🤖 2. Motor de IA Autónomo")
    tipo_modelo = st.selectbox("Seleccionar Algoritmo:", ["XGBoost", "CatBoost"])
    balancear = st.checkbox("Balanceo SMOTE (Casos Críticos)")

    st.divider()
    ejecutar = st.button("🚀 Iniciar Simulación Digital", use_container_width=True, type="primary")

if archivo is not None:
    df_raw = cargar_datos(archivo)
    
    if df_raw is not None:
        df_num = df_raw.select_dtypes(include=[np.number]).dropna()
        columnas = df_num.columns.tolist()
        
        with st.sidebar:
            st.header("🎯 3. Configuración de Variables")
            target = st.selectbox("Variable Objetivo (Y):", columnas, index=len(columnas)-1)
            features = st.multiselect("Predictores (X):", [c for c in columnas if c != target], 
                                     default=[c for c in columnas if c != target])

        if ejecutar or 'model' in st.session_state:
            if ejecutar:
                progress_bar = st.progress(0)
                # FASE 1: Limpieza
                df = df_num.copy()
                if modo_ruido == "Depuración por IQR":
                    Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
                    IQR = Q3 - Q1
                    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
                progress_bar.progress(25)

                # FASE 2: Dominios (UGM)
                best_k, best_score = 2, -1
                for k in range(2, 6):
                    if len(df) > k:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = km.fit_predict(df)
                        score = silhouette_score(df, labels)
                        if score > best_score: best_score, best_k = score, k
                kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                df['Dominio_GMD'] = kmeans_final.fit_predict(df)
                progress_bar.progress(50)

                # FASE 3: SMOTE (Híbrido)
                X, y = df[features], df[target]
                if balancear:
                    y_disc = pd.qcut(y, q=3, labels=False, duplicates='drop')
                    sm = SMOTE(random_state=42, k_neighbors=min(2, len(X)-1))
                    X_with_y = X.copy(); X_with_y['__target__'] = y
                    X_res, _ = sm.fit_resample(X_with_y, y_disc)
                    y, X = X_res['__target__'], X_res.drop(columns=['__target__'])
                progress_bar.progress(75)

                # FASE 4: Entrenamiento
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                if tipo_modelo == "XGBoost":
                    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_state=42, verbose=0)
                    model.fit(X_train, y_train, eval_set=(X_val, y_val))

                # FASE 5: Validación
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                y_pred_cv = cross_val_predict(model, X, y, cv=kf)
                
                st.session_state.model = model
                st.session_state.df_p = df
                st.session_state.y_pred = y_pred_cv
                st.session_state.metrics = (r2_score(y, y_pred_cv), mean_absolute_error(y, y_pred_cv), np.sqrt(mean_squared_error(y, y_pred_cv)), mean_absolute_percentage_error(y, y_pred_cv) * 100)
                st.session_state.X_f, st.session_state.y_f = X, y
                progress_bar.progress(100); time.sleep(0.5); progress_bar.empty()

            # Recuperación de datos
            model, df_p = st.session_state.model, st.session_state.df_p
            y_pred, y_f = st.session_state.y_pred, st.session_state.y_f
            r2, mae, rmse, mape = st.session_state.metrics
            X_f = st.session_state.X_f

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Datos", "📊 Variables", "🎯 Precisión", "🎛️ Simulador", "🚨 Monitor FDI"])

            with tab1:
                st.subheader("Caracterización de Unidades Geometalúrgicas (UGM)")
                st.dataframe(df_p.groupby('Dominio_GMD')[features + [target]].mean().style.background_gradient(cmap='viridis'))
                c1, c2 = st.columns(2)
                vx = c1.selectbox("Eje X:", df_p.columns, key="v_x")
                vy = c1.selectbox("Eje Y:", df_p.columns, index=columnas.index(target), key="v_y")
                if c1.button("🔄 Actualizar Gráfico"):
                    st.session_state.fig_exp = px.scatter(df_p, x=vx, y=vy, color='Dominio_GMD', trendline="ols") if vx != vy else px.histogram(df_p, x=vx, color='Dominio_GMD')
                if 'fig_exp' in st.session_state: c2.plotly_chart(st.session_state.fig_exp, use_container_width=True)

            with tab2:
                ch, ci = st.columns(2)
                ch.write("**Heatmap de Correlación**")
                ch.plotly_chart(px.imshow(df_p[[target] + features].corr(), text_auto=".2f", color_continuous_scale="RdBu_r"), use_container_width=True)
                ci.write("**Importancia de Variables (IA)**")
                imp = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.get_feature_importance()
                ci.plotly_chart(px.bar(pd.DataFrame({'V': features, 'I': imp}).sort_values('I'), x='I', y='V', orientation='h'), use_container_width=True)

            with tab3:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Fidelidad (R²)", f"{r2:.3f}"); m2.metric("MAE", f"{mae:.3f}")
                m3.metric("Riesgo (RMSE)", f"{rmse:.3f}"); m4.metric("Error Relativo", f"{mape:.2f}%")
                st.plotly_chart(px.scatter(x=y_f, y=y_pred, labels={'x': 'Real', 'y': 'Digital'}, trendline="ols"), use_container_width=True)

            with tab4:
                st.subheader("🎛️ Centro de Optimización Prescriptiva")
                # CORRECCIÓN DEL TypeError: Se añade el argumento '2'
                col_ctrl, col_res = st.columns(2)
                with col_ctrl:
                    st.info("🎮 **Ajuste Manual de Set-Points**")
                    inputs_sim = {col: st.slider(f"{col}", float(df_p[col].min()), float(df_p[col].max()), float(df_p[col].mean()), key=f"s_{col}") for col in features}
                    st.divider()
                    btn_opt = st.button("🚀 ENCONTRAR OPERACIÓN ÓPTIMA", use_container_width=True, type="primary")

                with col_res:
                    pred_manual = model.predict(pd.DataFrame([inputs_sim])).item()
                    if btn_opt:
                        rand_data = pd.DataFrame({c: np.random.uniform(df_p[c].min(), df_p[c].max(), 1000) for c in features})
                        preds_opt = model.predict(rand_data)
                        top_idx = np.argsort(preds_opt)[-5:][::-1]
                        st.session_state.top_5 = rand_data.iloc[top_idx].copy()
                        st.session_state.top_5['Recuperación_Estimada'] = preds_opt[top_idx]
                    
                    if 'top_5' in st.session_state:
                        # Selección de la mejor fila para el gráfico comparativo
                        mejor_cfg = st.session_state.top_5.iloc.to_dict()
                        mejor_val = mejor_cfg.pop('Recuperación_Estimada')
                        ganancia = mejor_val - pred_manual
                        
                        cont = st.container(border=True)
                        mc1, mc2 = cont.columns(2)
                        mc1.metric("Recuperación Actual", f"{pred_manual:.2f}%")
                        mc2.metric("Máximo Técnico", f"{mejor_val:.2f}%", delta=f"{ganancia:.2f}%")
                        
                        st.write("### 🥇 Top 5 Escenarios Recomendados")
                        st.dataframe(st.session_state.top_5.style.background_gradient(subset=['Recuperación_Estimada'], cmap='Blues'), use_container_width=True)
                        
                        fig_comp = go.Figure()
                        fig_comp.add_trace(go.Bar(name='Manual', x=features, y=[inputs_sim[f] for f in features]))
                        fig_comp.add_trace(go.Bar(name='Óptimo', x=features, y=[mejor_cfg[f] for f in features]))
                        fig_comp.update_layout(title="Comparativa: Manual vs Óptimo", barmode='group', height=350)
                        st.plotly_chart(fig_comp, use_container_width=True)

            with tab5:
                df_audit = X_f.copy(); df_audit[target], df_audit['Predicción'] = y_f, y_pred
                df_audit['Error'] = np.abs(df_audit[target] - df_audit['Predicción'])
                def sem(e): return "🟢 Normal" if e <= mae else ("🟡 Advertencia" if e <= 2*mae else "🔴 Anomalía")
                df_audit['Estado'] = df_audit['Error'].apply(sem)
                st.dataframe(df_audit[[target, 'Predicción', 'Error', 'Estado'] + features].head(500).style.map(
                    lambda x: "background-color: #90EE90" if x == "🟢 Normal" else ("background-color: #FFD700" if x == "🟡 Advertencia" else ("background-color: #F08080" if x == "🔴 Anomalía" else "")),
                    subset=['Estado']
                ))
        else:
            st.info("💡 Configure los parámetros y pulse 'Iniciar Simulación Digital'.")
else:
    st.info("👈 Cargue el dataset histórico para iniciar el Digital Twin.")
