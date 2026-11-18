"""Configuración global, constantes normativas (CONAGUA, 2016) y datos base de Tabasco."""

SEMILLA_ALEATORIA = 42

# Tabla 2.1. Coeficientes de escurrimiento por material de cubierta (CONAGUA, 2016)
COEFICIENTES_ESCURRIMIENTO = {
    "Techos metálicos o plásticos (PVC, polietileno)": 0.95,
    "Techos impermeabilizados o de materiales duros (teja)": 0.90,
    "Concreto hidráulico": 0.90,
    "Lámina metálica corrugada": 0.80,
}

# Precipitación media mensual histórica de referencia en Tabasco (mm/mes)
PRECIPITACION_MENSUAL_TABASCO = {
    1: ("Enero", 125.0),
    2: ("Febrero", 85.0),
    3: ("Marzo", 55.0),
    4: ("Abril", 45.0),
    5: ("Mayo", 95.0),
    6: ("Junio", 210.0),
    7: ("Julio", 190.0),
    8: ("Agosto", 225.0),
    9: ("Septiembre", 340.0),
    10: ("Octubre", 330.0),
    11: ("Noviembre", 185.0),
    12: ("Diciembre", 140.0),
}

DIAS_POR_MES = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}

# Intervalo de dotación de referencia OMS / CONAGUA (L/hab/día)
DOTACION_MIN_REFERENCIA = 50.0
DOTACION_MAX_REFERENCIA = 100.0
