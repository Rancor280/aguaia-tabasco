"""Fase 7: Pruebas unitarias de coherencia física, predictiva y validación de entradas."""

import joblib
import numpy as np
import pandas as pd
import pytest
from src.entrenar_modelo import entrenar_y_evaluar
from src.logica_hidrica import (
    calcular_balance_mensual,
    calcular_volumen_captable,
    generar_recomendaciones,
    validar_entradas,
)


def test_formula_volumen_captable_coincide_calculo_manual():
    v = calcular_volumen_captable(200.0, 100.0, "Concreto hidráulico")
    assert v == pytest.approx(18000.0)


def test_coherencia_volumen_crece_con_area_y_lluvia():
    v_base = calcular_volumen_captable(150.0, 60.0, "Lámina metálica corrugada")
    v_mayor_lluvia = calcular_volumen_captable(250.0, 60.0, "Lámina metálica corrugada")
    v_mayor_area = calcular_volumen_captable(150.0, 120.0, "Lámina metálica corrugada")
    assert v_mayor_lluvia > v_base
    assert v_mayor_area > v_base


def test_rechazo_entradas_invalidas():
    with pytest.raises(ValueError):
        validar_entradas(0, 80.0, "Concreto hidráulico", 6, 150.0)
    with pytest.raises(ValueError):
        validar_entradas(4, -20.0, "Concreto hidráulico", 6, 150.0)
    with pytest.raises(ValueError):
        validar_entradas(4, 80.0, "Concreto hidráulico", 6, -5.0)


def test_coherencia_modelo_consumo_crece_con_personas():
    entrenar_y_evaluar()
    modelo = joblib.load("models/modelo_consumo.joblib")
    base = {
        "area_techo_m2": 80.0,
        "precipitacion_mm": 200.0,
        "mes_sin": float(np.sin(2 * np.pi * 6 / 12)),
        "mes_cos": float(np.cos(2 * np.pi * 6 / 12)),
        "material_techo": "Concreto hidráulico",
    }
    df_test = pd.DataFrame([
        {"personas": 2, **base},
        {"personas": 6, **base},
    ])
    preds = modelo.predict(df_test)
    assert preds[1] > preds[0]


def test_aviso_potabilidad_siempre_presente():
    bal = calcular_balance_mensual(20000.0, 12000.0)
    recs = generar_recomendaciones(400.0, 4, bal["factor_satisfaccion"], 250.0)
    assert any("AVISO" in r and "consumo humano" in r for r in recs)
