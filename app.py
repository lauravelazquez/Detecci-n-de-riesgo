import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from flask import Flask, jsonify, request
from flask_cors import CORS
from staticmap import StaticMap, CircleMarker
from sklearn.neighbors import BallTree



app = Flask(__name__)
CORS(app)



MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "modelo_riesgo_zona.keras"
)

CLASSES_PATH = os.getenv(
    "CLASSES_PATH",
    "clases_riesgo_zona.json"
)

EVENTS_PATH = os.getenv(
    "EVENTS_PATH",
    "eventos_historicos.csv"
)



tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)



print(
    "Cargando clases...",
    flush=True
)

with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as archivo:

    CLASES = json.load(archivo)


print(
    "Clases:",
    CLASES,
    flush=True
)



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

print(
    "Entrada:",
    modelo.input_shape,
    flush=True
)

print(
    "Salida:",
    modelo.output_shape,
    flush=True
)



print(
    "Cargando eventos historicos...",
    flush=True
)

eventos = pd.read_csv(
    EVENTS_PATH
)


eventos.columns = [
    str(columna)
    .strip()
    .upper()

    for columna
    in eventos.columns
]


# Verificar columnas obligatorias

columnas_requeridas = {
    "LATITUD",
    "LONGITUD"
}


if not columnas_requeridas.issubset(
    eventos.columns
):

    raise ValueError(
        "eventos_historicos.csv debe contener "
        "las columnas LATITUD y LONGITUD."
    )


# Convertir coordenadas

eventos["LATITUD"] = pd.to_numeric(
    eventos["LATITUD"],
    errors="coerce"
)

eventos["LONGITUD"] = pd.to_numeric(
    eventos["LONGITUD"],
    errors="coerce"
)


# Eliminar registros sin coordenadas

eventos = (
    eventos
    .dropna(
        subset=[
            "LATITUD",
            "LONGITUD"
        ]
    )
    .copy()
)


# Validar rangos

eventos = eventos[
    eventos["LATITUD"].between(
        -90,
        90
    )
    &
    eventos["LONGITUD"].between(
        -180,
        180
    )
].copy()


print(
    "Eventos historicos cargados:",
    len(eventos),
    flush=True
)



RADIO_TIERRA_KM = 6371.0088


coordenadas_eventos = eventos[
    [
        "LATITUD",
        "LONGITUD"
    ]
].to_numpy(
    dtype=np.float64
)


coordenadas_eventos_rad = np.radians(
    coordenadas_eventos
)


arbol_eventos = BallTree(
    coordenadas_eventos_rad,
    metric="haversine"
)


print(
    "Indice espacial creado correctamente.",
    flush=True
)



def contar_eventos_radio(
    latitud,
    longitud,
    radio_km
):

    coordenada = np.radians(
        [[
            latitud,
            longitud
        ]]
    )


    indices = arbol_eventos.query_radius(
        coordenada,
        r=radio_km / RADIO_TIERRA_KM
    )[0]


    return int(
        len(indices)
    )



def obtener_conteos(
    latitud,
    longitud
):

    conteos = {

        "250m":
            contar_eventos_radio(
                latitud,
                longitud,
                0.25
            ),

        "500m":
            contar_eventos_radio(
                latitud,
                longitud,
                0.50
            ),

        "1km":
            contar_eventos_radio(
                latitud,
                longitud,
                1.00
            ),

        "2km":
            contar_eventos_radio(
                latitud,
                longitud,
                2.00
            )
    }


    print(
        "Eventos encontrados:",
        conteos,
        flush=True
    )


    return conteos



def generar_mapa(
    latitud,
    longitud
):

    print(
        "Generando mapa...",
        flush=True
    )


    # IMPORTANTE:
    # Debe coincidir con las imagenes de entrenamiento

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
        "Mapa generado.",
        flush=True
    )


    return imagen



def preparar_imagen(
    imagen
):

    print(
        "Preparando imagen...",
        flush=True
    )


    imagen = imagen.convert(
        "RGB"
    )


    arreglo = np.asarray(
        imagen,
        dtype=np.float32
    )


    # Durante el entrenamiento utilizaste 224x224

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
        "Min pixel:",
        float(
            entrada.min()
        ),
        flush=True
    )


    print(
        "Max pixel:",
        float(
            entrada.max()
        ),
        flush=True
    )


    return entrada



def obtener_protocolo(
    clase
):

    if clase == "ALTO":

        return (
            "ALERTA: Zona clasificada con alta "
            "concentracion historica de asaltos. "
            "Considerar monitoreo GPS y medidas "
            "adicionales de seguridad."
        )


    if clase == "MEDIO":

        return (
            "PRECAUCION: Zona clasificada con "
            "concentracion historica intermedia. "
            "Mantener monitoreo GPS activo."
        )


    if clase == "BAJO":

        return (
            "RIESGO RELATIVO BAJO: La zona presenta "
            "menor concentracion historica frente "
            "a las zonas utilizadas para entrenar "
            "el modelo."
        )


    return (
        "Nivel de riesgo no reconocido."
    )



def evaluar_ubicacion(
    latitud,
    longitud
):

    print(
        "",
        flush=True
    )

    print(
        "========================================",
        flush=True
    )

    print(
        f"Evaluando ubicacion: "
        f"{latitud}, {longitud}",
        flush=True
    )



    conteos = obtener_conteos(
        latitud,
        longitud
    )



    imagen = generar_mapa(
        latitud,
        longitud
    )



    entrada = preparar_imagen(
        imagen
    )



    print(
        "Ejecutando modelo...",
        flush=True
    )


    salida = modelo(
        entrada,
        training=False
    )


    probabilidades = np.asarray(
        salida
    )[0]


    print(
        "Probabilidades:",
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
        probabilidades[
            indice
        ]
    )


    print(
        "Riesgo:",
        clase,
        flush=True
    )


    print(
        "Confianza:",
        confianza,
        flush=True
    )



    desglose = {}


    for posicion, nombre_clase in enumerate(
        CLASES
    ):

        desglose[
            nombre_clase
        ] = round(
            float(
                probabilidades[
                    posicion
                ]
            ) * 100,
            2
        )



    resultado = {

        "coordenadas": [
            round(
                latitud,
                8
            ),
            round(
                longitud,
                8
            )
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


        "eventos_historicos": {

            "250m":
                conteos["250m"],

            "500m":
                conteos["500m"],

            "1km":
                conteos["1km"],

            "2km":
                conteos["2km"]
        },


        "protocolo":
            obtener_protocolo(
                clase
            )
    }


    print(
        "Resultado final:",
        resultado,
        flush=True
    )


    return resultado



@app.route(
    "/",
    methods=["GET"]
)
def inicio():

    return jsonify({

        "servicio":
            "API de evaluacion de riesgo por zona",

        "estado":
            "activo",

        "modelo":
            MODEL_PATH,

        "clases":
            CLASES,

        "eventos_historicos_cargados":
            len(eventos)
    })



@app.route(
    "/salud",
    methods=["GET"]
)
def salud():

    return jsonify({

        "estado":
            "ok",

        "modelo":
            "cargado",

        "eventos":
            len(eventos)
    })



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

        "forma_entrada":
            str(
                modelo.input_shape
            ),

        "forma_salida":
            str(
                modelo.output_shape
            ),

        "eventos_historicos":
            len(eventos),

        "radios":
            [
                "250m",
                "500m",
                "1km",
                "2km"
            ]
    })



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


        # ====================================================
        # EJECUTAR EVALUACION
        # ====================================================

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
    ) as error:

        print(
            "Error de coordenadas:",
            str(error),
            flush=True
        )


        return jsonify({

            "error":
                "Latitud y longitud son obligatorias "
                "y deben ser numericas."

        }), 400


    except Exception as error:

        print(
            "ERROR DURANTE LA PREDICCION:",
            str(error),
            flush=True
        )


        app.logger.exception(
            error
        )


        return jsonify({

            "error":
                "No fue posible evaluar la ubicacion."

        }), 500



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
