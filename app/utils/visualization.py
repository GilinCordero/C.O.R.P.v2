"""Plotly visualization utilities for the C.O.R.P. hourly app."""
import plotly.graph_objects as go
import pandas as pd

DIA_SEMANA = {0: "Lunes", 1: "Martes", 2: "Miercoles", 3: "Jueves", 4: "Viernes", 5: "Sabado", 6: "Domingo"}

COLOR_PREDICTION = "#1f77b4"
COLOR_HISTORICAL = "#d62728"
COLOR_BARS = "#3498db"
COLOR_SECONDARY = "#ff7f0e"
TEMPLATE = "plotly_white"


def create_hourly_forecast_chart(historical_df: pd.DataFrame, forecast_df: pd.DataFrame, plant_name: str, days_back: int = 7):
    """Line chart showing historical hourly volume + hourly forecast."""
    fig = go.Figure()

    # Historical (last N days)
    if historical_df is not None and not historical_df.empty:
        hist = historical_df.copy()
        hist["hour_bucket"] = pd.to_datetime(hist["hour_bucket"])
        hist = hist.sort_values("hour_bucket")
        hist["dia_semana"] = hist["hour_bucket"].dt.dayofweek.map(DIA_SEMANA)
        fig.add_trace(go.Scatter(
            x=hist["hour_bucket"],
            y=hist["volume_m3"],
            mode="lines",
            name="Historico (ultimos dias)",
            line=dict(color=COLOR_HISTORICAL, width=1.5),
            customdata=hist[["dia_semana"]],
            hovertemplate="%{x|%Y-%m-%d %H:%M} (%{customdata[0]})<br>Real: %{y:.2f} m3<extra></extra>",
        ))

    # Forecast
    fc = forecast_df.copy()
    fc["hour_bucket"] = pd.to_datetime(fc["hour_bucket"])
    fc = fc.sort_values("hour_bucket")
    fc["dia_semana"] = fc["hour_bucket"].dt.dayofweek.map(DIA_SEMANA)

    # Confidence band P10-P90
    has_band = "lower" in fc.columns and "upper" in fc.columns
    if has_band:
        fig.add_trace(go.Scatter(
            x=fc["hour_bucket"],
            y=fc["upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fc["hour_bucket"],
            y=fc["lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            name="Intervalo P10-P90",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=fc["hour_bucket"],
        y=fc["predicted_hourly_m3"],
        mode="lines",
        name="Prediccion",
        line=dict(color=COLOR_PREDICTION, width=2),
        customdata=fc[["dia_semana"]],
        hovertemplate="%{x|%Y-%m-%d %H:%M} (%{customdata[0]})<br>Prediccion: %{y:.2f} m3<extra></extra>",
    ))

    fig.update_layout(
        template=TEMPLATE,
        title=f"Prediccion Horaria - {plant_name}",
        xaxis_title="Fecha y Hora",
        yaxis_title="Volumen (m3/hora)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig


def create_hourly_validation_chart(val_df: pd.DataFrame, plant_id: int, plant_name: str, sample_limit: int = 500):
    """Line chart of actual vs predicted on test set (hourly)."""
    sub = val_df[val_df["ship_plant_code"] == plant_id].copy()
    sub["hour_bucket"] = pd.to_datetime(sub["hour_bucket"])
    sub = sub.sort_values("hour_bucket")

    # Downsample for display if too large
    if len(sub) > sample_limit:
        step = len(sub) // sample_limit
        sub = sub.iloc[::step].copy()

    fig = go.Figure()
    sub["dia_semana"] = sub["hour_bucket"].dt.dayofweek.map(DIA_SEMANA)

    fig.add_trace(go.Scatter(
        x=sub["hour_bucket"],
        y=sub["volume_m3"],
        mode="lines",
        name="Real",
        line=dict(color=COLOR_HISTORICAL, width=1.5),
        customdata=sub[["dia_semana"]],
        hovertemplate="%{x|%Y-%m-%d %H:%M} (%{customdata[0]})<br>Real: %{y:.2f} m3<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=sub["hour_bucket"],
        y=sub["predicted"],
        mode="lines",
        name="Prediccion",
        line=dict(color=COLOR_PREDICTION, width=1.5),
        customdata=sub[["dia_semana"]],
        hovertemplate="%{x|%Y-%m-%d %H:%M} (%{customdata[0]})<br>Prediccion: %{y:.2f} m3<extra></extra>",
    ))

    fig.update_layout(
        template=TEMPLATE,
        title=f"Validacion Horaria - {plant_name}",
        xaxis_title="Fecha y Hora",
        yaxis_title="Volumen (m3/hora)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig


def create_daily_aggregation_chart(forecast_df: pd.DataFrame, plant_name: str):
    """Aggregate hourly forecast to daily and show bar chart."""
    fc = forecast_df.copy()
    fc["hour_bucket"] = pd.to_datetime(fc["hour_bucket"])
    fc["date"] = fc["hour_bucket"].dt.date
    daily = fc.groupby("date")["predicted_hourly_m3"].sum().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["dia_semana"] = daily["date"].dt.dayofweek.map(DIA_SEMANA)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily["date"],
        y=daily["predicted_hourly_m3"],
        marker_color=COLOR_BARS,
        name="Volumen Diario (agregado)",
        customdata=daily[["dia_semana"]],
        hovertemplate="%{x|%Y-%m-%d} (%{customdata[0]})<br>Total: %{y:.1f} m3<extra></extra>",
    ))

    fig.update_layout(
        template=TEMPLATE,
        title=f"Agregacion Diaria - {plant_name}",
        xaxis_title="Fecha",
        yaxis_title="Volumen (m3/dia)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig
