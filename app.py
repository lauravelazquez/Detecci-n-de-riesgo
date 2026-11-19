import os

import numpy as np
import tensorflow as tf

from flask import Flask, jsonify, request
from flask_cors import CORS
from staticmap import StaticMap, CircleMarker


# ============================================================
# CONFIGURACIÓN DE FLASK
# ============================================================

app = Flask(__name__)

# Permitir llamadas desde GitHub Pages
CORS(app)


# ============================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "modelo_riesgo.keras"
)

CLASES = [
    "ALTO",
    "BAJO",
    "MEDIO"
]

UMBRAL = 0.65


# ============================================================
# CARGAR MODELO
# ============================================================

print("Cargando modelo...", flush=True)

modelo = tf.keras.models.load_model(
    MODEL_PATH
)

print("Modelo cargado correctamente.", flush=True)


# ============================================================
# PREPROCESAMIENTO
# ============================================================

preprocesamiento = tf.keras.Sequential([
    tf.keras.layers.CenterCrop(
        height=360,
        width=540
    ),

    tf.keras.layers.Resizing(
        224,
        224
    ),

    tf.keras.layers.Rescaling(
        1. / 127.5,
        offset=-1
    )
])


# ============================================================
# EVALUACIÓN DE UNA UBICACIÓN
# ============================================================

def evaluar_ubicacion(latitud, longitud):

    print(
        f"1. Iniciando evaluación: {latitud}, {longitud}",
        flush=True
    )

    # --------------------------------------------------------
    # GENERAR MAPA
    # --------------------------------------------------------

    mapa = StaticMap(
        600,
        400
    )

    mapa.add_marker(
        CircleMarker(
            (longitud, latitud),
            "#FF0000",
            12
        )
    )

    print(
        "2. Antes de generar mapa",
        flush=True
    )

    imagen = mapa.render(
        zoom=17
    )

    print(
        "3. Mapa generado correctamente",
        flush=True
    )


    # --------------------------------------------------------
    # CONVERTIR IMAGEN
    # --------------------------------------------------------

    arreglo = tf.keras.utils.img_to_array(
        imagen
    )

    lote = tf.expand_dims(
        arreglo,
        0
    )

    print(
        "4. Imagen convertida a tensor",
        flush=True
    )


    # --------------------------------------------------------
    # PREPROCESAMIENTO
    # --------------------------------------------------------

    entrada = preprocesamiento(
        lote
    )

    print(
        "5. Preprocesamiento terminado",
        flush=True
    )


    # --------------------------------------------------------
    # PREDICCIÓN
    # --------------------------------------------------------

    print(
        "6. Antes de modelo.predict()",
        flush=True
    )

    pred = modelo.predict(
        entrada,
        verbose=0
    )[0]

    print(
        "7. Predicción terminada",
        flush=True
    )

    print(
        f"Probabilidades obtenidas: {pred}",
        flush=True
    )


    # --------------------------------------------------------
    # POSTPROCESAMIENTO
    # --------------------------------------------------------

    indice = int(
        np.argmax(pred)
    )

    clase = CLASES[
        indice
    ]

    confianza = float(
        pred[indice]
    )

    print(
        f"8. Clase calculada: {clase}",
        flush=True
    )

    print(
        f"9. Confianza: {confianza}",
        flush=True
    )


    # --------------------------------------------------------
    # PROTOCOLO
    # --------------------------------------------------------

    if confianza < UMBRAL:

        estado = (
            f"REVISIÓN MANUAL "
            f"(Posible {clase})"
        )

        protocolo = (
            "El modelo no está suficientemente seguro. "
            "Validar la ruta con un analista."
        )

    elif clase == "ALTO":

        estado = clase

        protocolo = (
            "ALERTA: Asignar custodia o "
            "cambiar el horario de entrega."
        )

    elif clase == "MEDIO":

        estado = clase

        protocolo = (
            "PRECAUCIÓN: Mantener monitoreo "
            "GPS activo."
        )

    else:

        estado = clase

        protocolo = (
            "RUTA SEGURA: Operación normal."
        )


    # --------------------------------------------------------
    # RESPUESTA
    # --------------------------------------------------------

    resultado = {

        "coordenadas": [
            latitud,
            longitud
        ],

        "riesgo_detectado":
            estado,

        "confianza_porcentaje":
            round(
                confianza * 100,
                2
            ),

        "desglose_probabilidades": {

            CLASES[i]:
                round(
                    float(pred[i]) * 100,
                    2
                )

            for i in range(
                len(CLASES)
            )
        },

        "protocolo":
            protocolo
    }

    print(
        "10. Evaluación terminada correctamente",
        flush=True
    )

    return resultado


# ============================================================
# ENDPOINT PRINCIPAL
# ============================================================

@app.get("/")
def inicio():

    return jsonify({

        "servicio":
            "API de evaluación de riesgo",

        "estado":
            "activo"
    })


# ============================================================
# ENDPOINT DE SALUD
# ============================================================

@app.get("/salud")
def salud():

    return jsonify({
        "estado": "ok"
    })


# ============================================================
# ENDPOINT DE PREDICCIÓN
# ============================================================

@app.post("/predecir")
def predecir():

    print(
        ">>> Petición recibida en /predecir",
        flush=True
    )

    try:

        datos = (
            request.get_json(
                silent=True
            )
            or {}
        )

        print(
            f">>> Datos recibidos: {datos}",
            flush=True
        )

        lat = float(
            datos.get(
                "latitud"
            )
        )

        lon = float(
            datos.get(
                "longitud"
            )
        )


        # Validar coordenadas

        if not (
            -90 <= lat <= 90
            and
            -180 <= lon <= 180
        ):

            return jsonify({

                "error":
                    "Coordenadas fuera de rango."

            }), 400


        print(
            ">>> Coordenadas válidas",
            flush=True
        )


        resultado = evaluar_ubicacion(
            lat,
            lon
        )


        print(
            ">>> Enviando respuesta al navegador",
            flush=True
        )


        return jsonify(
            resultado
        )


    except (
        TypeError,
        ValueError
    ) as exc:

        print(
            f"ERROR DE DATOS: {exc}",
            flush=True
        )

        return jsonify({

            "error":
                "Latitud y longitud son obligatorias "
                "y deben ser numéricas."

        }), 400


    except Exception as exc:

        print(
            f"ERROR DURANTE INFERENCIA: {exc}",
            flush=True
        )

        app.logger.exception(
            exc
        )

        return jsonify({

            "error":
                "No fue posible procesar la ubicación."

  =========================================================
# EJECUCIÓN LOCAL
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
