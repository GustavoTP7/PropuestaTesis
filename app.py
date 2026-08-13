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
import matplotlib.pyplot as plt
import time

# --- CONFIGURACIÓN DE INTERFAZ ---
st.set_page_config(page_title="Geomet Twin Pro", layout="wide")

@st.cache_data
def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        df.columns = df.columns.astype(str).str.strip()
        df = df.loc[:, ~df.columns.duplicated()] # Limpieza de duplicados
        return df
    except Exception as e:
        st.error(f"Error en la ingesta de datos: {e}")
        return None

st.title("💎 Geomet Twin Pro: Inteligencia Operacional")
st.markdown("**Digital Twin de Soporte a la Decisión (DSS)**. Estructura persistente: solo se reinicia si tú lo decides.")

# --- BARRA LATERAL (ENTRENAMIENTO GLOBAL) ---
with st.sidebar:
    st.header("⚙️ 1. Configuración del Modelo")
    archivo = st.file_uploader("Subir registros históricos", type=["csv", "xlsx"])
    modo_ruido = st.radio("Filtro de Outliers [IQR]:", ["Data Original", "Depuración por IQR"])
    tipo_modelo = st.selectbox("Algoritmo:", ["XGBoost", "CatBoost"])
    balancear = st.checkbox("Balanceo SMOTE (Casos Críticos)")

    st.divider()
    # Botón ÚNICO para entrenar: Al presionarlo, se genera el Gemelo Digital
    ejecutar = st.button("🚀 Iniciar Simulación Digital", use_container_width=True)

if archivo is not None:
    df_raw = cargar_datos(archivo)
    
    if df_raw is not None:
        df_num = df_raw.select_dtypes(include=[np.number]).dropna()
        columnas = df_num.columns.tolist()
        
        with st.sidebar:
            st.header("🎯 2. Variables de Proceso")
            target = st.selectbox("Variable Objetivo (Y):", columnas, index=len(columnas)-1)
            features = st.multiselect("Predictores (X):", [c for c in columnas if c != target], 
                                     default=[c for c in columnas if c != target])

        # --- LÓGICA DE PERSISTENCIA ---
        # Si se presiona el botón o ya existe un modelo en memoria, entramos a las pestañas
        if ejecutar or 'model' in st.session_state:
            if ejecutar:
                # FASE 1: Limpieza e IQR
                df = df_num.copy()
                if modo_ruido == "Depuración por IQR":
                    Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
                    IQR = Q3 - Q1
                    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

                # FASE 2: Dominios Inteligentes (Clusters) [5]
                best_k, best_score = 2, -1
                for k in range(2, 6):
                    if len(df) > k:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = km.fit_predict(df)
                        score = silhouette_score(df, labels)
                        if score > best_score: best_score, best_k = score, k
                kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                df['Dominio_GMD'] = kmeans_final.fit_predict(df)

                # FASE 3: Balanceo SMOTE (Evitando KeyError de índices)
                X, y = df[features], df[target]
                if balancear:
                    y_disc = pd.qcut(y, q=3, labels=False, duplicates='drop')
                    sm = SMOTE(random_state=42, k_neighbors=min(2, len(X)-1))
                    X_with_y = X.copy(); X_with_y['__target__'] = y
                    X_res, _ = sm.fit_resample(X_with_y, y_disc)
                    y, X = X_res['__target__'], X_res.drop(columns=['__target__'])

                # FASE 4: Entrenamiento
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                if tipo_modelo == "XGBoost":
                    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_state=42, verbose=0)
                    model.fit(X_train, y_train, eval_set=(X_val, y_val))

                # FASE 5: Guardar en Session State para que no se borre al cambiar pestañas
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                y_pred_cv = cross_val_predict(model, X, y, cv=kf)
                
                st.session_state.model = model
                st.session_state.df_p = df
                st.session_state.y_pred = y_pred_cv
                st.session_state.metrics = (r2_score(y, y_pred_cv), mean_absolute_error(y, y_pred_cv), np.sqrt(mean_squared_error(y, y_pred_cv)), mean_absolute_percentage_error(y, y_pred_cv) * 100)
                st.session_state.X_f, st.session_state.y_f = X, y

            # Recuperar datos guardados
            model, df_p = st.session_state.model, st.session_state.df_p
            y_pred, y_f = st.session_state.y_pred, st.session_state.y_f
            r2, mae, rmse, mape = st.session_state.metrics
            X_f = st.session_state.X_f

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Datos & Clusters", "📊 Multivariante", "🎯 Precisión", "🎛️ Simulador", "🚨 Auditoría FDI"])

            with tab1:
                st.subheader("Caracterización de Unidades Geometalúrgicas (UGM)")
                st.dataframe(df_p.groupby('Dominio_GMD')[features + [target]].mean().style.background_gradient(cmap='viridis'))
                
                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    vx = st.selectbox("Eje X:", df_p.columns, key="v_x")
                    vy = st.selectbox("Eje Y:", df_p.columns, index=columnas.index(target), key="v_y")
                    # BOTÓN LOCAL: Al cambiar variables aquí, NO se reinicia el modelo
                    if st.button("🔄 Actualizar Gráfico"):
                        if vx == vy: st.session_state.fig_t1 = px.histogram(df_p, x=vx, color='Dominio_GMD')
                        else: st.session_state.fig_t1 = px.scatter(df_p, x=vx, y=vy, color='Dominio_GMD', trendline="ols")
                with c2:
                    if 'fig_t1' in st.session_state: st.plotly_chart(st.session_state.fig_t1, use_container_width=True)

            with tab2:
                c_h, c_i = st.columns(2)
                c_h.write("**Heatmap de Interdependencia**")
                c_h.plotly_chart(px.imshow(df_p[[target] + features].corr(), text_auto=".2f", color_continuous_scale="RdBu_r"), use_container_width=True)
                c_i.write("**Importancia Relativa de Variables**")
                imp = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.get_feature_importance()
                c_i.plotly_chart(px.bar(pd.DataFrame({'V': features, 'I': imp}).sort_values('I'), x='I', y='V', orientation='h'), use_container_width=True)

            with tab3:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Fidelidad (R²)", f"{r2:.3f}"); m2.metric("MAE", f"{mae:.3f}"); m3.metric("RMSE", f"{rmse:.3f}"); m4.metric("Error Relativo", f"{mape:.2f}%")
                st.plotly_chart(px.scatter(x=y_f, y=y_pred, labels={'x': 'Realidad', 'y': 'Digital Twin'}, trendline="ols", title="Ajuste Real vs Digital"), use_container_width=True)

            with tab4:
                st.subheader("Simulación Prescriptiva")
                cs1, cs2 = st.columns([6, 7])
                with cs1:
                    inputs_sim = {col: st.slider(f"{col}:", float(df_p[col].min()), float(df_p[col].max()), float(df_p[col].mean()), key=f"s_{col}") for col in features}
                    st.divider()
                    # BOTÓN LOCAL PARA OPTIMIZAR
                    if st.button("🚀 Optimizar Operación"):
                        rand_data = pd.DataFrame({c: np.random.uniform(df_p[c].min(), df_p[c].max(), 500) for c in features})
                        preds = model.predict(rand_data)
                        idx = np.argmax(preds)
                        st.session_state.opt_cfg = rand_data.iloc[idx].to_dict()
                        st.session_state.opt_val = preds[idx]
                with cs2:
                    pred_manual = model.predict(pd.DataFrame([inputs_sim])).item()
                    st.metric(label=f"Predicción {target} (Manual)", value=f"{pred_manual:.2f}%")
                    if 'opt_cfg' in st.session_state:
                        st.success(f"✅ Máximo Técnico Encontrado: {st.session_state.opt_val:.2f}%")
                        st.json(st.session_state.opt_cfg)

            with tab5:
                st.subheader("Monitor FDI (Auditoría de Turnos)")
                df_audit = X_f.copy(); df_audit[target], df_audit['Predicción'] = y_f, y_pred
                df_audit['Error'] = np.abs(df_audit[target] - df_audit['Predicción'])
                def sem(e): return "🟢 Normal" if e <= mae else "🟡 Advertencia" if e <= 2*mae else "🔴 Anomalía"
                df_audit['Estado'] = df_audit['Error'].apply(sem)
                st.dataframe(df_audit[[target, 'Predicción', 'Error', 'Estado'] + features].head(1000).style.map(
                    lambda x: "background-color: #90EE90" if x == "🟢 Normal" else ("background-color: #FFD700" if x == "🟡 Advertencia" else ("background-color: #F08080" if x == "🔴 Anomalía" else "")),
                    subset=['Estado']
                ))
        else:
            st.info("💡 Configure los parámetros en la barra lateral y presione 'Iniciar Simulación Digital'.")
else:
    st.info("👈 Cargue el dataset histórico para iniciar el Digital Twin.")
