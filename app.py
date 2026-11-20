import os
import json

import numpy as np
import tensorflow as tf

from flask import Flask, jsonify, request
from flask_cors import CORS
from staticmap import StaticMap, CircleMarker


# ============================================================
# CONFIGURACION DE FLASK
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# ARCHIVOS DEL MODELO
# ============================================================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "modelo_riesgo_zona.keras"
)

CLASSES_PATH = os.getenv(
    "CLASSES_PATH",
    "clases_riesgo_zona.json"
)

CONFIG_PATH = os.getenv(
    "CONFIG_PATH",
    "configuracion_riesgo_zona.json"
)


# ============================================================
# REDUCIR CONSUMO DE RECURSOS EN RENDER
# ============================================================

tf.config.threading.set_inter_op_parallelism_threads(1)

tf.config.threading.set_intra_op_parallelism_threads(1)


# ============================================================
# CARGAR CLASES
# ============================================================

with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as archivo:

    CLASES = json.load(
        archivo
    )


print(
    "Clases cargadas:",
    CLASES,
    flush=True
)


# ============================================================
# CARGAR CONFIGURACION
# ============================================================

if os.path.exists(CONFIG_PATH):

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as archivo:

        CONFIGURACION = json.load(
            archivo
        )

else:

    CONFIGURACION = {}


print(
    "Configuracion:",
    CONFIGURACION,
    flush=True
)


# ============================================================
# CARGAR MODELO
# ============================================================

print(
    "Cargando modelo de riesgo por zona...",
    flush=True
)

modelo = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "Modelo cargado correctamente.",
    flush=True
)

print(
    "Entrada del modelo:",
    modelo.input_shape,
    flush=True
)

print(
    "Salida del modelo:",
    modelo.output_shape,
    flush=True
)


# ============================================================
# GENERAR IMAGEN DEL MAPA
# ============================================================

def generar_imagen_mapa(
    latitud,
    longitud
):

    print(
        "1. Generando imagen cartografica...",
        flush=True
    )

    # Debe coincidir con el entrenamiento
    mapa = StaticMap(
        244,
        244
    )

    marcador = CircleMarker(
        (
            longitud,
            latitud
        ),
        "#FF0000",
        7
    )

    mapa.add_marker(
        marcador
    )

    imagen = mapa.render(
        zoom=16
    )

    print(
        "2. Imagen generada.",
        flush=True
    )

    return imagen


# ============================================================
# PREPARAR IMAGEN PARA EL MODELO
# ============================================================

def preparar_imagen(
    imagen
):

    print(
        "3. Preparando imagen...",
        flush=True
    )

    # La imagen ya se generó en 244 x 244
    imagen = imagen.convert(
        "RGB"
    )

    arreglo = np.asarray(
        imagen,
        dtype=np.float32
    )

    # El modelo fue entrenado con image_size=(224, 224).
    # Redimensionamos únicamente, sin normalizar aquí.
    arreglo = tf.image.resize(
        arreglo,
        (
            224,
            224
        )
    )

    entrada = np.expand_dims(
        arreglo.numpy(),
        axis=0
    )

    print(
        "Forma de entrada:",
        entrada.shape,
        flush=True
    )

    print(
        "Rango de pixeles:",
        float(entrada.min()),
        float(entrada.max()),
        flush=True
    )

    return entrada


# ============================================================
# DEFINIR PROTOCOLO
# ============================================================

def obtener_protocolo(
    clase
):

    if clase == "ALTO":

        return (
            "ALERTA: Zona clasificada con alta "
            "concentracion historica de asaltos. "
            "Considerar cambio de horario, monitoreo GPS "
            "y medidas adicionales de seguridad."
        )

    if clase == "MEDIO":

        return (
            "PRECAUCION: Zona con concentracion historica "
            "intermedia. Mantener monitoreo GPS activo."
        )

    return (
        "RIESGO RELATIVO BAJO: La zona presenta menor "
        "concentracion historica frente a las zonas "
        "utilizadas para entrenar el modelo."
    )


# ============================================================
# EVALUAR UBICACION
# ============================================================

def evaluar_ubicacion(
    latitud,
    longitud
):

    print(
        f"Evaluando: {latitud}, {longitud}",
        flush=True
    )

    imagen = generar_imagen_mapa(
        latitud,
        longitud
    )

    entrada = preparar_imagen(
        imagen
    )

    print(
        "4. Ejecutando modelo...",
        flush=True
    )

    # Evita el overhead de model.predict()
    salida = modelo(
        entrada,
        training=False
    )

    probabilidades = np.asarray(
        salida
    )[0]

    print(
        "5. Probabilidades:",
        probabilidades,
        flush=True
    )

    indice = int(
        np.argmax(
            probabilidades
        )
    )

    clase = CLASES[
        indice
    ]

    confianza = float(
        probabilidades[indice]
    )

    desglose = {}

    for posicion, nombre_clase in enumerate(
        CLASES
    ):

        desglose[
            nombre_clase
        ] = round(
            float(
                probabilidades[posicion]
            ) * 100,
            2
        )

    resultado = {
        "coordenadas": [
            round(latitud, 8),
            round(longitud, 8)
        ],

        "riesgo_detectado":
            clase,

        "confianza_porcentaje":
            round(
                confianza * 100,
                2
            ),

        "desglose_probabilidades":
            desglose,

        "protocolo":
            obtener_protocolo(
                clase
            ),

        "interpretacion": (
            "El resultado representa el nivel relativo "
            "de concentracion historica de asaltos "
            "aprendido por el modelo visual."
        )
    }

    print(
        "6. Resultado:",
        resultado,
        flush=True
    )

    return resultado


# ============================================================
# RUTA PRINCIPAL
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def inicio():

    return jsonify({
        "servicio":
            "API de riesgo por zona",

        "estado":
            "activo",

        "modelo":
            MODEL_PATH,

        "clases":
            CLASES
    })


# ============================================================
# RUTA DE SALUD
# ============================================================

@app.route(
    "/salud",
    methods=["GET"]
)
def salud():

    return jsonify({
        "estado":
            "ok",

        "modelo":
            "cargado"
    })


# ============================================================
# RUTA DE DIAGNOSTICO
# ============================================================

@app.route(
    "/diagnostico",
    methods=["GET"]
)
def diagnostico():

    return jsonify({
        "modelo":
            MODEL_PATH,

        "clases":
            CLASES,

        "entrada":
            str(
                modelo.input_shape
            ),

        "salida":
            str(
                modelo.output_shape
            ),

        "configuracion":
            CONFIGURACION
    })


# ============================================================
# RUTA DE PREDICCION
# ============================================================

@app.route(
    "/predecir",
    methods=["POST"]
)
def predecir():

    print(
        "Peticion recibida en /predecir",
        flush=True
    )

    try:

        datos = request.get_json(
            silent=True
        ) or {}

        latitud = float(
            datos.get(
                "latitud"
            )
        )

        longitud = float(
            datos.get(
                "longitud"
            )
        )

        if not (
            -90 <= latitud <= 90
            and
            -180 <= longitud <= 180
        ):

            return jsonify({
                "error":
                    "Coordenadas fuera de rango."
            }), 400

        resultado = evaluar_ubicacion(
            latitud,
            longitud
        )

        return jsonify(
            resultado
        )

    except (
        TypeError,
        ValueError
    ):

        return jsonify({
            "error":
                "Latitud y longitud son obligatorias "
                "y deben ser numericas."
        }), 400

    except Exception as error:

        print(
            "ERROR:",
            str(error),
            flush=True
        )

        app.logger.exception(
            error
        )

        return jsonify({
            "error":
                "No fue posible evaluar la zona."
        }), 500


# ============================================================
# EJECUCION LOCAL
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                5000
            )
        ),
        debug=False
    )
