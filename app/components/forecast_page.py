"""Forecast page: hourly predictions + daily aggregation."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import (
    load_hourly_features,
    load_validation,
    load_metrics,
    get_plant_names,
)
from utils.model_runner import run_hourly_forecast, save_forecast_csv
from utils.visualization import (
    create_hourly_forecast_chart,
    create_daily_aggregation_chart,
    create_hourly_validation_chart,
)

MAX_FORECAST_DAYS = 45


def show_forecast_page():
    st.title("Prediccion de Demanda de Concreto")
    st.caption("C.O.R.P. v2")

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    st.sidebar.header("Configuracion")

    plant_names = get_plant_names()
    plant_options = {v: k for k, v in plant_names.items()}
    selected_name = st.sidebar.selectbox("Selecciona Planta", list(plant_options.keys()))
    plant_id = plant_options[selected_name]

    st.sidebar.markdown("---")
    metrics = load_metrics()
    st.sidebar.metric("MAE Horario (Test)", f"{metrics['mae_hourly_m3']:.2f} m3/hr")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = load_hourly_features()
    val_df = load_validation()

    plant_hist = df[df["ship_plant_code"] == plant_id].copy()
    last_time = plant_hist["hour_bucket"].max()
    if isinstance(last_time, str):
        last_time = pd.to_datetime(last_time)

    forecast_start = last_time
    st.info(f"Datos actualizados al {last_time.strftime('%Y-%m-%d %H:%M')}. Prediciendo desde la siguiente hora.")

    # Slider outside tabs so it's available everywhere
    days = st.slider("Dias a predecir", min_value=1, max_value=MAX_FORECAST_DAYS, value=7, key=f"days_slider_{plant_id}")
    hours = days * 24

    # Compute forecast once and cache in session_state
    cache_key = f"forecast_{plant_id}_{days}"
    if cache_key not in st.session_state:
        with st.spinner("Generando prediccion horaria..."):
            st.session_state[cache_key] = run_hourly_forecast(plant_id, forecast_start, hours)

    forecast = st.session_state[cache_key]

    # Add P10-P90 confidence band based on historical errors (full test set)
    plant_val_errors = val_df[
        val_df["ship_plant_code"] == plant_id
    ]["error"].values
    if len(plant_val_errors) > 0:
        p10_err = np.percentile(plant_val_errors, 10)
        p90_err = np.percentile(plant_val_errors, 90)
        forecast["lower"] = (forecast["predicted_hourly_m3"] + p10_err).clip(lower=0)
        forecast["upper"] = forecast["predicted_hourly_m3"] + p90_err
    else:
        mae_global = metrics["mae_hourly_m3"]
        forecast["lower"] = (forecast["predicted_hourly_m3"] - 1.64 * mae_global).clip(lower=0)
        forecast["upper"] = forecast["predicted_hourly_m3"] + 1.64 * mae_global

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["Prediccion Horaria", "Agregacion Diaria", "Validacion del Modelo"])

    # ==================== TAB 1: Hourly Forecast ====================
    with tab1:
        col1, col2, col3 = st.columns(3)
        total_pred = forecast["predicted_hourly_m3"].sum()
        avg_pred = forecast["predicted_hourly_m3"].mean()
        max_pred = forecast["predicted_hourly_m3"].max()

        col1.metric("Total (prox. dias)", f"{total_pred:,.0f} m3")
        col2.metric("Promedio Horario", f"{avg_pred:.2f} m3")
        col3.metric("Hora Pico", f"{max_pred:.2f} m3")

        # Chart: last 7 days history + forecast
        cutoff = forecast_start - timedelta(days=7)
        hist_7d = plant_hist[plant_hist["hour_bucket"] >= cutoff].copy()
        fig_hourly = create_hourly_forecast_chart(hist_7d, forecast, plant_names[plant_id])
        st.plotly_chart(fig_hourly, use_container_width=True)

        # Table
        st.subheader("Tabla de Predicciones Horarias")
        display_df = forecast.copy()
        display_df["hour_bucket"] = display_df["hour_bucket"].dt.strftime("%Y-%m-%d %H:%M")
        display_df = display_df.rename(columns={
            "hour_bucket": "Fecha y Hora",
            "predicted_hourly_m3": "Volumen Predicho (m3)",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Export
        csv_buffer = io.StringIO()
        forecast.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Descargar CSV",
            data=csv_buffer.getvalue(),
            file_name=f"forecast_hourly_{plant_id}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="export_hourly",
        )

    # ==================== TAB 2: Daily Aggregation ====================
    with tab2:
        st.info("Agregacion de la prediccion horaria a nivel diario.")

        fig_daily = create_daily_aggregation_chart(forecast, plant_names[plant_id])
        st.plotly_chart(fig_daily, use_container_width=True)

        st.subheader("Tabla de Agregacion Diaria")
        fc = forecast.copy()
        fc["hour_bucket"] = pd.to_datetime(fc["hour_bucket"])
        fc["date"] = fc["hour_bucket"].dt.date
        daily = fc.groupby("date")["predicted_hourly_m3"].sum().reset_index()
        daily = daily.rename(columns={
            "date": "Fecha",
            "predicted_hourly_m3": "Volumen Total (m3)",
        })
        daily["Fecha"] = daily["Fecha"].astype(str)
        st.dataframe(daily, use_container_width=True, hide_index=True)

    # ==================== TAB 3: Validation ====================
    with tab3:
        st.info("Comparacion entre valores reales y predichos en el conjunto de prueba.")

        plant_val = val_df[val_df["ship_plant_code"] == plant_id].copy()
        if not plant_val.empty:
            mae = plant_val["error"].abs().mean()
            st.metric("MAE Horario (Test)", f"{mae:.2f} m3")

            fig_val = create_hourly_validation_chart(val_df, plant_id, plant_names[plant_id])
            st.plotly_chart(fig_val, use_container_width=True)
        else:
            st.warning("No hay datos de validacion para esta planta.")
