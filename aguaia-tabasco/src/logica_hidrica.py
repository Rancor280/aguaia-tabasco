"""Fases 2, 6 y 8: Cálculo de potencial pluvial (V = P × A × C), balance y recomendaciones."""

import numpy as np
import pandas as pd
from src.config import (
    COEFICIENTES_ESCURRIMIENTO,
    DIAS_POR_MES,
    DOTACION_MAX_REFERENCIA,
    DOTACION_MIN_REFERENCIA,
    PRECIPITACION_MENSUAL_TABASCO,
)


def validar_entradas(personas: int, area_techo_m2: float, material_techo: str, mes: int, precipitacion_mm: float) -> None:
    if personas is None or int(personas) <= 0:
        raise ValueError("El número de personas debe ser un entero mayor a 0.")
    if area_techo_m2 is None or float(area_techo_m2) <= 0:
        raise ValueError("El área del techo debe ser mayor a 0 m².")
    if material_techo not in COEFICIENTES_ESCURRIMIENTO:
        raise ValueError(f"Material de techo no válido: {material_techo}")
    if mes not in range(1, 13):
        raise ValueError("El mes debe estar entre 1 y 12.")
    if precipitacion_mm is None or float(precipitacion_mm) < 0:
        raise ValueError("La precipitación no puede ser negativa.")


def calcular_volumen_captable(precipitacion_mm: float, area_techo_m2: float, material_techo: str) -> float:
    """Calcula V = P × A × C en litros/mes según lineamientos de CONAGUA (2016)."""
    if area_techo_m2 <= 0 or precipitacion_mm < 0:
        raise ValueError("Área debe ser > 0 y precipitación >= 0.")
    coef_c = COEFICIENTES_ESCURRIMIENTO[material_techo]
    return round(float(precipitacion_mm) * float(area_techo_m2) * coef_c, 2)


def calcular_balance_mensual(volumen_captable_l: float, consumo_mensual_l: float) -> dict:
    """Calcula balance neto (L/mes), % de cobertura y factor de satisfacción Fs."""
    if consumo_mensual_l <= 0:
        raise ValueError("El consumo mensual debe ser mayor a 0.")
    balance_l = volumen_captable_l - consumo_mensual_l
    factor_satisfaccion = volumen_captable_l / consumo_mensual_l
    porcentaje_cobertura = min(100.0, factor_satisfaccion * 100.0)
    return {
        "balance_l": round(balance_l, 2),
        "factor_satisfaccion": round(factor_satisfaccion, 2),
        "porcentaje_cobertura": round(porcentaje_cobertura, 1),
    }


def generar_recomendaciones(
    consumo_diario_l: float,
    personas: int,
    factor_satisfaccion: float,
    precipitacion_mm: float,
) -> list[str]:
    """Emite recomendaciones trazables según las reglas definidas en la Tabla 3.3."""
    recomendaciones = []
    dotacion_persona = consumo_diario_l / personas

    if dotacion_persona > DOTACION_MAX_REFERENCIA:
        recomendaciones.append(
            f"Consumo elevado ({dotacion_persona:.1f} L/hab/día > 100 L): revisar fugas, reducir tiempos de ducha, "
            "evitar riego con manguera e instalar dispositivos ahorradores."
        )
    elif dotacion_persona >= DOTACION_MIN_REFERENCIA:
        recomendaciones.append(
            f"Consumo dentro del intervalo de referencia ({dotacion_persona:.1f} L/hab/día, rango 50–100 L): "
            "mantener hábitos actuales y monitorear periódicamente."
        )
    else:
        recomendaciones.append(
            f"Consumo austero ({dotacion_persona:.1f} L/hab/día < 50 L): verificar que se cubran las necesidades "
            "básicas de higiene del hogar."
        )

    if factor_satisfaccion >= 1.0:
        recomendaciones.append(
            f"La captación pluvial cubre la demanda mensual estimada (Fs = {factor_satisfaccion:.2f} ≥ 1): "
            "el agua recolectada podría abastecer usos no potables (sanitarios, limpieza y riego) durante este mes."
        )
    else:
        recomendaciones.append(
            f"La captación cubre parcialmente la demanda mensual (Fs = {factor_satisfaccion:.2f} < 1): "
            "priorizar usos de menor exigencia de calidad y evaluar un tanque de almacenamiento de mayor capacidad."
        )

    if precipitacion_mm < 100.0 and factor_satisfaccion < 1.0:
        recomendaciones.append(
            "Mes de baja precipitación con déficit hídrico: planear el almacenamiento de excedentes "
            "durante los meses más lluviosos (junio a octubre)."
        )

    recomendaciones.append(
        "AVISO DE ALCANCE Y SALUD: El agua de lluvia captada requiere filtración, desinfección y verificación "
        "de calidad antes de cualquier uso para consumo humano. Esta herramienta es informativa."
    )
    return recomendaciones


def simular_perfil_anual(modelo, personas: int, area_techo_m2: float, material_techo: str) -> pd.DataFrame:
    """Genera la proyección de los 12 meses para visualización comparativa."""
    filas = []
    for mes in range(1, 13):
        nombre_mes, precip_mm = PRECIPITACION_MENSUAL_TABASCO[mes]
        dias = DIAS_POR_MES[mes]
        entrada = pd.DataFrame([{
            "personas": personas,
            "area_techo_m2": area_techo_m2,
            "precipitacion_mm": precip_mm,
            "mes_sin": float(np.sin(2 * np.pi * mes / 12)),
            "mes_cos": float(np.cos(2 * np.pi * mes / 12)),
            "material_techo": material_techo,
        }])
        consumo_diario = float(modelo.predict(entrada)[0])
        consumo_mensual = round(consumo_diario * dias, 2)
        volumen_captable = calcular_volumen_captable(precip_mm, area_techo_m2, material_techo)
        filas.append({
            "Mes_num": mes,
            "Mes": nombre_mes,
            "Precipitación (mm)": precip_mm,
            "Consumo estimado (L)": consumo_mensual,
            "Volumen captable (L)": volumen_captable,
        })
    return pd.DataFrame(filas)
