"""Fases 5 y 7: Entrenamiento, validación cruzada, evaluación y persistencia del modelo."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from src.config import SEMILLA_ALEATORIA
from src.generar_datos import generar_dataset

ATRIBUTOS_NUMERICOS = ["personas", "area_techo_m2", "precipitacion_mm", "mes_sin", "mes_cos"]
ATRIBUTOS_CATEGORICOS = ["material_techo"]
OBJETIVO = "consumo_diario_l"


def entrenar_y_evaluar(
    ruta_datos: Path = Path("data/dataset_aguaia_tabasco.csv"),
    dir_modelos: Path = Path("models"),
) -> dict:
    if not ruta_datos.exists():
        ruta_datos.parent.mkdir(parents=True, exist_ok=True)
        df = generar_dataset()
        df.to_csv(ruta_datos, index=False, encoding="utf-8")
    else:
        df = pd.read_csv(ruta_datos)

    X = df[ATRIBUTOS_NUMERICOS + ATRIBUTOS_CATEGORICOS]
    y = df[OBJETIVO]
    grupos = df["vivienda_id"]

    # División 80/20 agrupada por vivienda para evitar fuga de datos
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEMILLA_ALEATORIA)
    idx_train, idx_test = next(splitter.split(X, y, groups=grupos))

    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    grupos_train = grupos.iloc[idx_train]

    preprocesador_lineal = ColumnTransformer([
        ("num", StandardScaler(), ATRIBUTOS_NUMERICOS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ATRIBUTOS_CATEGORICOS),
    ])

    preprocesador_arbol = ColumnTransformer([
        ("num", "passthrough", ATRIBUTOS_NUMERICOS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ATRIBUTOS_CATEGORICOS),
    ])

    modelos = {
        "Referencia (Promedio)": Pipeline([
            ("prep", preprocesador_arbol),
            ("reg", DummyRegressor(strategy="mean")),
        ]),
        "Regresión Lineal Múltiple": Pipeline([
            ("prep", preprocesador_lineal),
            ("reg", LinearRegression()),
        ]),
        "Árbol de Decisión": Pipeline([
            ("prep", preprocesador_arbol),
            ("reg", DecisionTreeRegressor(max_depth=8, random_state=SEMILLA_ALEATORIA)),
        ]),
    }

    # Ajuste de hiperparámetros con validación cruzada de 5 pliegues para Random Forest
    rf_base = Pipeline([
        ("prep", preprocesador_arbol),
        ("reg", RandomForestRegressor(random_state=SEMILLA_ALEATORIA, n_jobs=-1)),
    ])
    param_grid = {
        "reg__n_estimators": [100, 200],
        "reg__max_depth": [8, 12, None],
        "reg__min_samples_leaf": [2, 5],
    }
    cv = GroupKFold(n_splits=5)
    busqueda_rf = GridSearchCV(
        rf_base,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    busqueda_rf.fit(X_train, y_train, groups=grupos_train)
    modelos["Random Forest"] = busqueda_rf.best_estimator_

    resultados = {}
    for nombre, pipeline in modelos.items():
        if nombre != "Random Forest":
            pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        mse = float(mean_squared_error(y_test, pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, pred))
        resultados[nombre] = {
            "MAE_L_dia": round(mae, 2),
            "MSE_L2": round(mse, 2),
            "RMSE_L_dia": round(rmse, 2),
            "R2": round(r2, 4),
        }

    # Persistir el mejor modelo seleccionado (Random Forest)
    dir_modelos.mkdir(parents=True, exist_ok=True)
    ruta_modelo = dir_modelos / "modelo_consumo.joblib"
    joblib.dump(modelos["Random Forest"], ruta_modelo)

    salida_metricas = {
        "mejores_hiperparametros_rf": busqueda_rf.best_params_,
        "metricas_prueba": resultados,
    }
    (dir_modelos / "metricas_evaluacion.json").write_text(
        json.dumps(salida_metricas, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return salida_metricas


if __name__ == "__main__":
    res = entrenar_y_evaluar()
    print(json.dumps(res, indent=2, ensure_ascii=False))
