"""Fase 3: Generación y depuración del conjunto de datos sintéticos parametrizados."""

from pathlib import Path
import numpy as np
import pandas as pd
from src.config import (
    COEFICIENTES_ESCURRIMIENTO,
    PRECIPITACION_MENSUAL_TABASCO,
    SEMILLA_ALEATORIA,
)


def generar_dataset(n_viviendas: int = 500, semilla: int = SEMILLA_ALEATORIA) -> pd.DataFrame:
    """Genera registros mensuales de viviendas simuladas en Tabasco."""
    rng = np.random.default_rng(semilla)
    materiales = list(COEFICIENTES_ESCURRIMIENTO.keys())
    registros = []

    for vivienda_id in range(1, n_viviendas + 1):
        # Distribución de habitantes centrada alrededor de ~4 personas (referencia INEGI)
        personas = int(np.clip(rng.normal(loc=3.9, scale=1.5), 1, 10))
        # Área de techo correlacionada parcialmente con tamaño del hogar
        area_techo = float(np.round(np.clip(rng.normal(loc=45 + personas * 10, scale=18), 25, 220), 1))
        material = str(rng.choice(materiales, p=[0.20, 0.25, 0.35, 0.20]))
        coef_c = COEFICIENTES_ESCURRIMIENTO[material]

        # Dotación base por persona dentro del rango 50 - 115 L/hab/día
        dotacion_hogar = rng.uniform(52.0, 112.0)

        for mes in range(1, 13):
            _, precip_base = PRECIPITACION_MENSUAL_TABASCO[mes]
            precipitacion = float(np.round(max(5.0, rng.normal(loc=precip_base, scale=precip_base * 0.15)), 1))

            # Factor estacional suave (mayor demanda en meses cálidos de primavera/verano)
            factor_estacional = 1.0 + 0.08 * np.sin(2 * np.pi * (mes - 3) / 12)
            # Efecto climático limitado y no lineal sobre el consumo (Xenochristou, 2019)
            ajuste_lluvia = -0.02 * (precipitacion / 350.0)
            ruido = rng.normal(loc=0.0, scale=12.0)

            consumo_diario = personas * dotacion_hogar * (factor_estacional + ajuste_lluvia) + ruido
            consumo_diario = float(np.round(max(personas * 35.0, consumo_diario), 2))

            registros.append({
                "vivienda_id": vivienda_id,
                "personas": personas,
                "area_techo_m2": area_techo,
                "material_techo": material,
                "coeficiente_c": coef_c,
                "mes": mes,
                "mes_sin": round(float(np.sin(2 * np.pi * mes / 12)), 6),
                "mes_cos": round(float(np.cos(2 * np.pi * mes / 12)), 6),
                "precipitacion_mm": precipitacion,
                "consumo_diario_l": consumo_diario,
            })

    df = pd.DataFrame(registros)
    return depurar_dataset(df)


def depurar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina nulos, duplicados, rangos inválidos y valores atípicos extremos (IQR)."""
    df = df.dropna().drop_duplicates()
    df = df[
        (df["personas"] > 0)
        & (df["area_techo_m2"] > 0)
        & (df["precipitacion_mm"] >= 0)
        & (df["consumo_diario_l"] > 0)
    ].copy()

    q1 = df["consumo_diario_l"].quantile(0.25)
    q3 = df["consumo_diario_l"].quantile(0.75)
    iqr = q3 - q1
    limite_inf = max(0.0, q1 - 1.5 * iqr)
    limite_sup = q3 + 1.5 * iqr
    df = df[(df["consumo_diario_l"] >= limite_inf) & (df["consumo_diario_l"] <= limite_sup)].reset_index(drop=True)
    return df


if __name__ == "__main__":
    Path("data").mkdir(parents=True, exist_ok=True)
    dataset = generar_dataset()
    ruta_salida = Path("data/dataset_aguaia_tabasco.csv")
    dataset.to_csv(ruta_salida, index=False, encoding="utf-8")
    print(f"Dataset generado correctamente: {len(dataset)} registros en {ruta_salida}")
