# Detección y clasificación de zonas de riesgo

## Objetivo

Clasificación de zonas de riesgo a vendedores en ruta de ser asaltados en la Ciudad de México y Estado de México

## Modelo

Se utiliza un modelo de clasificación previamente entrenado y
almacenado en:

modelo_riesgo.keras

El modelo clasifica las ubicaciones de acuerdo con su nivel de riesgo en cuanto a la afectación economica del histórico de eventos del 2024, 2025 y 2026.

## Pipeline de inferencia

El proyecto implementa las siguientes etapas:

1. Carga del modelo entrenado.
2. Preprocesamiento de los datos de entrada.
3. Ejecución del modelo.
4. Obtención de la predicción.
5. Postprocesamiento de resultados.
6. Inferencia por lote.
7. Simulación de inferencia en tiempo real.

## Inferencia por lote

El sistema permite procesar múltiples ubicaciones y generar
los resultados correspondientes a cada ruta.

Los resultados se almacenan en:

Rutas_Evaluadas_Seguridad.xlsx

## Inferencia en tiempo real

Se implementó una simulación de monitoreo en tiempo real,
en la cual se reciben ubicaciones y se ejecuta el pipeline
de inferencia para identificar el nivel de riesgo.

## Archivos

- `Detección.ipynb`: notebook con el pipeline completo.
- `modelo_riesgo.keras`: modelo entrenado.
- `Asaltos 1 (2).xlsx`: conjunto de datos utilizado.
- `Rutas_Evaluadas_Seguridad.xlsx`: resultados de inferencia por lote.

## Librerias

- Python
- TensorFlow / Keras
- MobileNetV2
- Pandas
- NumPy
- Google Colab

## Resultado

Se construyó un pipeline funcional que permite cargar el modelo
entrenado, preparar los datos, realizar predicciones y procesar
ubicaciones tanto por lote como mediante una simulación en
tiempo real.
