"""Data management page: model info, CSV upload, retraining guidance."""
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.data_loader import load_metrics


def show_data_management_page():
    st.title("Gestion de Datos y Modelo")
    st.caption("C.O.R.P. v2 - Estado del sistema")

    # ------------------------------------------------------------------
    # Model metrics
    # ------------------------------------------------------------------
    st.header("Metricas del Modelo")
    metrics = load_metrics()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE Horario", f"{metrics['mae_hourly_m3']:.2f} m3/hr")
    col2.metric("RMSE", f"{metrics['rmse_hourly_m3']:.2f} m3/hr")
    col3.metric("sMAPE", f"{metrics['smape_pct']:.1f}%")
    col4.metric("Filas Validacion", f"{metrics.get('validation_rows', '—'):,}")

    st.markdown("---")

    # ------------------------------------------------------------------
    # CSV Upload
    # ------------------------------------------------------------------
    st.header("Subir Nuevos Datos")
    uploaded = st.file_uploader(
        "Sube un archivo CSV con remisiones actualizadas",
        type=["csv"],
        help="El CSV debe contener las mismas columnas que el dataset original.",
    )

    if uploaded is not None:
        try:
            df_new = pd.read_csv(uploaded)
            st.success(f"Archivo cargado: {len(df_new):,} filas, {len(df_new.columns)} columnas.")
            st.write("Vista previa:")
            st.dataframe(df_new.head(), use_container_width=True)
            st.info(
                "Para integrar estos datos al modelo, descarga el CSV y corre el pipeline "
                "de reentrenamiento localmente (ver instrucciones abajo)."
            )
        except Exception as e:
            st.error(f"Error al leer el CSV: {e}")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Retraining instructions
    # ------------------------------------------------------------------
    st.header("Reentrenamiento del Modelo")

    st.warning(
        "**Limitacion de Streamlit Cloud:** El reentrenamiento debe hacerse **localmente**. "
        "Streamlit Cloud no permite entrenar modelos ni guardar archivos grandes de forma persistente.",
        icon="⚠️",
    )

    st.subheader("Pasos para reentrenar manualmente")
    st.markdown(
        """
        1. Actualiza el archivo Excel de remisiones en `Modulo_Alberto/data/processed/`.
        2. Corre el script de feature engineering:
           ```bash
           python app/scripts/build_hourly_dataset.py
           ```
        3. Corre el script de entrenamiento:
           ```bash
           python app/scripts/train_hourly_model.py
           ```
        4. Reemplaza los archivos generados en `app/models/` y `app/data/`.
        5. Reinicia la app de Streamlit.
        """
    )
