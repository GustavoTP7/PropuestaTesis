import numpy as np
import pandas as pd


def parse_shift_report(df_raw):
    """Procesa y limpia la estructura de reportes geometalúrgicos detallados por turno."""
    df = df_raw.copy()

    # Normalizar nombres de columnas eliminando espacios y caracteres especiales
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
        .str.replace("%", "PCT")
    )

    # 1. Feature Engineering Mineralógico
    # Ratio de Sulfuros Secundarios vs Primarios (Cct + Cv) / (Cpy + Bn)
    if "CHALCOPYRITE" in df.columns and "CHALCOCITE" in df.columns:
        cpy = df["CHALCOPYRITE"].fillna(0)
        cct = df["CHALCOCITE"].fillna(0)
        cv = df.get("COVELLITE", 0)
        df["RATIO_SECONDARY_CU"] = (cct + cv) / np.where(
            cpy == 0, 0.001, cpy
        )

    # Contenido Total de Arcillas (Sericita + Caolín + Clorita)
    clay_cols = [c for c in ["SERICITE", "KAOLINITE", "ILLITE"] if c in df.columns]
    if clay_cols:
        df["TOTAL_CLAYS_PCT"] = df[clay_cols].sum(axis=1)

    # Ratio Pirita / Cobre Total (Indicador de Selectividad en Flotación)
    if "PYRITE" in df.columns and "CUT" in df.columns:
        df["PYRITE_CUT_RATIO"] = df["PYRITE"] / np.where(
            df["CUT"] == 0, 0.001, df["CUT"]
        )

    # 2. Feature Engineering de Solubilidad (Leyes)
    if "CUS" in df.columns and "CUT" in df.columns:
        df["RATIO_CUS_CUT"] = df["CUS"] / np.where(df["CUT"] == 0, 0.001, df["CUT"])

    if "CUCN" in df.columns and "CUT" in df.columns:
        df["RATIO_CUCN_CUT"] = df["CUCN"] / np.where(
            df["CUT"] == 0, 0.001, df["CUT"]
        )

    # Cálculo del Cobre Residual (Insoluble / Calcopirita - Enargita)
    if all(col in df.columns for col in ["CUT", "CUS", "CUCN"]):
        df["CURES_PCT"] = (df["CUT"] - df["CUS"] - df["CUCN"]).clip(lower=0)

    return df
