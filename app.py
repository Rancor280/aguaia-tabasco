"""Fase 6: Interfaz de usuario de AguaIA Tabasco desarrollada con Streamlit."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.config import (
    COEFICIENTES_ESCURRIMIENTO,
    DIAS_POR_MES,
    PRECIPITACION_MENSUAL_TABASCO,
)
from src.entrenar_modelo import entrenar_y_evaluar
from src.logica_hidrica import (
    calcular_balance_mensual,
    calcular_volumen_captable,
    generar_recomendaciones,
    simular_perfil_anual,
    validar_entradas,
)

st.set_page_config(page_title="AguaIA Tabasco", page_icon="💧", layout="wide")


@st.cache_resource
def cargar_modelo():
    ruta = Path("models/modelo_consumo.joblib")
    if not ruta.exists():
        entrenar_y_evaluar()
    return joblib.load(ruta)


modelo = cargar_modelo()

st.title("💧 AguaIA Tabasco")
st.subheader("Estimación de consumo doméstico y potencial de captación pluvial por vivienda")
st.info(
    "**Aviso de alcance:** Prototipo académico informativo. Los cálculos de captación siguen los "
    "Lineamientos Técnicos de CONAGUA (2016), no sustituyen estudios de ingeniería hidráulica ni "
    "determinan la potabilidad del agua."
)

with st.sidebar:
    st.header("Datos de la vivienda")
    personas = st.number_input("Número de habitantes", min_value=1, max_value=20, value=4, step=1)
    area_techo = st.number_input("Área de techo proyectada (m²)", min_value=10.0, max_value=500.0, value=80.0, step=5.0)
    material_techo = st.selectbox("Material de la cubierta", list(COEFICIENTES_ESCURRIMIENTO.keys()))
    mes = st.selectbox(
        "Mes de análisis",
        options=list(PRECIPITACION_MENSUAL_TABASCO.keys()),
        format_func=lambda m: f"{m} - {PRECIPITACION_MENSUAL_TABASCO[m][0]}",
        index=8,
    )
    _, precip_default = PRECIPITACION_MENSUAL_TABASCO[mes]
    precipitacion = st.number_input(
        "Precipitación mensual (mm)",
        min_value=0.0,
        max_value=1200.0,
        value=float(precip_default),
        step=5.0,
        help="Valor precargado con media histórica de Tabasco; puedes ajustarlo manualmente.",
    )
    consumo_real_opcional = st.number_input(
        "Consumo diario real opcional (L/día) [0 = omitir]",
        min_value=0.0,
        max_value=10000.0,
        value=0.0,
        step=10.0,
    )

try:
    validar_entradas(personas, area_techo, material_techo, mes, precipitacion)

    entrada_modelo = pd.DataFrame([{
        "personas": personas,
        "area_techo_m2": area_techo,
        "precipitacion_mm": precipitacion,
        "mes_sin": float(np.sin(2 * np.pi * mes / 12)),
        "mes_cos": float(np.cos(2 * np.pi * mes / 12)),
        "material_techo": material_techo,
    }])

    consumo_diario_est = float(modelo.predict(entrada_modelo)[0])
    dias_mes = DIAS_POR_MES[mes]
    consumo_mensual_est = consumo_diario_est * dias_mes
    volumen_captable = calcular_volumen_captable(precipitacion, area_techo, material_techo)
    balance = calcular_balance_mensual(volumen_captable, consumo_mensual_est)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Consumo diario estimado", f"{consumo_diario_est:,.1f} L/día", f"{consumo_diario_est / personas:,.1f} L/hab/día")
    col2.metric("Demanda mensual estimada", f"{consumo_mensual_est:,.0f} L/mes")
    col3.metric("Volumen pluvial captable", f"{volumen_captable:,.0f} L/mes", f"C = {COEFICIENTES_ESCURRIMIENTO[material_techo]}")
    col4.metric("Cobertura pluvial", f"{balance['porcentaje_cobertura']:.1f} %", f"Fs = {balance['factor_satisfaccion']:.2f}")

    if consumo_real_opcional > 0:
        diferencia = consumo_real_opcional - consumo_diario_est
        st.caption(
            f"Comparación con tu consumo real ingresado ({consumo_real_opcional:,.1f} L/día): "
            f"diferencia de {diferencia:+,.1f} L/día respecto a la estimación del modelo."
        )

    st.divider()
    st.subheader("Proyección anual: Demanda estimada vs. Potencial de captación")
    df_anual = simular_perfil_anual(modelo, personas, area_techo, material_techo)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_anual["Mes"], y=df_anual["Consumo estimado (L)"], name="Consumo estimado (L)", marker_color="#1f77b4"))
    fig.add_trace(go.Bar(x=df_anual["Mes"], y=df_anual["Volumen captable (L)"], name="Volumen captable (L)", marker_color="#2ca02c"))
    fig.update_layout(barmode="group", xaxis_title="Mes", yaxis_title="Litros / mes", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recomendaciones personalizadas")
    recomendaciones = generar_recomendaciones(
        consumo_diario_l=consumo_diario_est,
        personas=personas,
        factor_satisfaccion=balance["factor_satisfaccion"],
        precipitacion_mm=precipitacion,
    )
    for rec in recomendaciones:
        if "AVISO" in rec:
            st.warning(rec)
        else:
            st.success(rec)

except ValueError as err:
    st.error(f"Error de validación: {err}")
