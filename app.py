import os
import numpy as np
import tensorflow as tf

from flask import Flask, jsonify, request
from flask_cors import CORS
from staticmap import StaticMap, CircleMarker
from PIL import Image


# ============================================================
# CONFIGURACION
# ============================================================

app = Flask(__name__)

CORS(app)

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
# CONFIGURACION DE TENSORFLOW
# ============================================================

# Reducir uso de CPU/hilos en Render
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


# ============================================================
# CARGAR MODELO
# ============================================================

print(
    "Cargando modelo...",
    flush=True
)

modelo = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "Modelo cargado correctamente.",
    flush=True
)


# ============================================================
# PREPROCESAMIENTO CON PIL + NUMPY
# ============================================================

def preparar_imagen(imagen):

    print(
        "4. Iniciando preprocesamiento con PIL",
        flush=True
    )

    # Asegurar RGB
    imagen = imagen.convert("RGB")

    ancho, alto = imagen.size

    # --------------------------------------------------------
    # CENTER CROP EQUIVALENTE A 540 x 360
    # --------------------------------------------------------

    crop_width = min(
        540,
        ancho
    )

    crop_height = min(
        360,
        alto
    )

    izquierda = (
        ancho - crop_width
    ) // 2

    superior = (
        alto - crop_height
    ) // 2

    derecha = (
        izquierda + crop_width
    )

    inferior = (
        superior + crop_height
    )

    imagen = imagen.crop(
        (
            izquierda,
            superior,
            derecha,
            inferior
        )
    )

    print(
        "5. Center crop terminado",
        flush=True
    )


    # --------------------------------------------------------
    # RESIZE 224 x 224
    # --------------------------------------------------------

    imagen = imagen.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    print(
        "6. Resize terminado",
        flush=True
    )


    # --------------------------------------------------------
    # CONVERTIR A NUMPY
    # --------------------------------------------------------

    entrada = np.asarray(
        imagen,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # NORMALIZACION -1 A 1
    # Equivalente a Rescaling(1./127.5, offset=-1)
    # --------------------------------------------------------

    entrada = (
        entrada / 127.5
    ) - 1.0


    # Agregar dimension batch
    entrada = np.expand_dims(
        entrada,
        axis=0
    )


    print(
        f"7. Entrada preparada: {entrada.shape}",
        flush=True
    )

    return entrada


# ============================================================
# EVALUAR UBICACION
# ============================================================

def evaluar_ubicacion(
    latitud,
    longitud
):

    print(
        f"1. Iniciando evaluacion: "
        f"{latitud}, {longitud}",
        flush=True
    )


    # ========================================================
    # GENERAR MAPA
    # ========================================================

    mapa = StaticMap(
        600,
        400
    )

    mapa.add_marker(
        CircleMarker(
            (
                longitud,
                latitud
            ),
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


    # ========================================================
    # PREPROCESAMIENTO
    # ========================================================

    entrada = preparar_imagen(
        imagen
    )


    # ========================================================
    # PREDICCION
    # ========================================================

    print(
        "8. Antes de ejecutar el modelo",
        flush=True
    )

    # Ejecutar directamente el modelo.
    # Evitamos model.predict() para reducir overhead.
    salida = modelo(
        entrada,
        training=False
    )

    print(
        "9. Modelo ejecutado",
        flush=True
    )


    # Convertir salida TensorFlow a NumPy
    pred = np.asarray(
        salida
    )[0]


    print(
        f"10. Probabilidades: {pred}",
        flush=True
    )


    # ========================================================
    # POSTPROCESAMIENTO
    # ========================================================

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
        f"11. Clase: {clase}",
        flush=True
    )

    print(
        f"12. Confianza: {confianza}",
        flush=True
    )


    # ========================================================
    # PROTOCOLO
    # ========================================================

    if confianza < UMBRAL:

        estado = (
            f"REVISION MANUAL "
            f"(Posible {clase})"
        )

        protocolo = (
            "El modelo no esta suficientemente seguro. "
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
            "PRECAUCION: Mantener "
            "monitoreo GPS activo."
        )


    else:

        estado = clase

        protocolo = (
            "RUTA SEGURA: Operacion normal."
        )


    # ========================================================
    # PROBABILIDADES
    # ========================================================

    desglose = {}

    for i in range(
        len(CLASES)
    ):

        desglose[
            CLASES[i]
        ] = round(
            float(pred[i]) * 100,
            2
        )


    # ========================================================
    # RESULTADO
    # ========================================================

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

        "desglose_probabilidades":
            desglose,

        "protocolo":
            protocolo
    }


    print(
        "13. Evaluacion terminada correctamente",
        flush=True
    )


    return resultado


# ============================================================
# API PRINCIPAL
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def inicio():

    return jsonify({

        "servicio":
            "API de evaluacion de riesgo",

        "estado":
            "activo"
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/salud",
    methods=["GET"]
)
def salud():

    return jsonify({
        "estado": "ok"
    })


# ============================================================
# PREDICCION
# ============================================================

@app.route(
    "/predecir",
    methods=["POST"]
)
def predecir():

    print(
        ">>> Peticion recibida en /predecir",
        flush=True
    )

    try:

        datos = request.get_json(
            silent=True
        ) or {}


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


        # ====================================================
        # VALIDACION
        # ====================================================

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
            ">>> Coordenadas validas",
            flush=True
        )


        # ====================================================
        # INFERENCIA
        # ====================================================

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
                "Latitud y longitud son "
                "obligatorias y deben ser numericas."

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
                "No fue posible procesar "
                "la ubicacion."

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
