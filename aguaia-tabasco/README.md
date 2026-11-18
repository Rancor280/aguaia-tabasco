# AguaIA Tabasco

Prototipo de sistema inteligente para estimar el consumo doméstico de agua y el potencial de captación pluvial (SCALL) en viviendas del estado de Tabasco, México.

## Aviso de alcance
Este sistema es un prototipo académico de carácter informativo y educativo. Los cálculos de captación pluvial siguen los *Lineamientos Técnicos: Sistema de captación de agua de lluvia con fines de abasto de agua potable a nivel vivienda* (CONAGUA, 2016), **no sustituyen estudios de ingeniería hidráulica** y **no determinan la potabilidad del agua recolectada**.

## Estructura del repositorio

```text
aguaia-tabasco/
├── data/
│   └── dataset_aguaia_tabasco.csv   # Conjunto de datos sintéticos parametrizados y depurados
├── models/
│   ├── modelo_consumo.joblib        # Pipeline Random Forest entrenado y persistido
│   └── metricas_evaluacion.json     # Métricas MAE, MSE, RMSE y R² en prueba
├── src/
│   ├── __init__.py
│   ├── config.py                    # Parámetros normativos (Tabla 2.1) y lluvias de Tabasco
│   ├── generar_datos.py             # Fase 3: Generación y depuración del dataset
│   ├── entrenar_modelo.py           # Fases 5 y 7: Entrenamiento, validación cruzada y evaluación
│   └── logica_hidrica.py            # Fases 2, 6 y 8: Fórmula V = P × A × C, balance y recomendaciones
├── tests/
│   └── test_sistema.py              # Fase 7: Pruebas automatizadas con pytest
├── app.py                           # Fase 6: Interfaz web interactiva en Streamlit
├── requirements.txt                 # Dependencias del proyecto
└── README.md
```

## Instalación y ejecución

1. Clonar el repositorio e instalar dependencias:
   ```bash
   git clone https://github.com/Rancor280/aguaia-tabasco.git
   cd aguaia-tabasco
   pip install -r requirements.txt
   ```

2. Generar el conjunto de datos y entrenar el modelo (opcional, ya se incluyen generados):
   ```bash
   python -m src.generar_datos
   python -m src.entrenar_modelo
   ```

3. Ejecutar pruebas automatizadas:
   ```bash
   pytest
   ```

4. Iniciar la interfaz de usuario:
   ```bash
   streamlit run app.py
   ```
