import os
import json
from io import BytesIO

import matplotlib

# Render no tiene una pantalla gráfica.
# Agg permite generar las imágenes sin interfaz.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.neighbors import BallTree

app = Flask(__name__)

CORS(app)


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "modelo_riesgo_densidad.keras"
)

CLASSES_PATH = os.getenv(
    "CLASSES_PATH",
    "clases_riesgo_densidad.json"
)

CONFIG_PATH = os.getenv(
    "CONFIG_PATH",
    "configuracion_riesgo_densidad.json"
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

    CLASES = json.load(
        archivo
    )


print(
    "Clases cargadas:",
    CLASES,
    flush=True
)


if os.path.exists(
    CONFIG_PATH
):

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


RADIO_VISUAL_KM = float(
    CONFIGURACION.get(
        "radio_visual_km",
        2.0
    )
)


print(
    "Configuracion cargada:",
    CONFIGURACION,
    flush=True
)

print(
    "Radio visual:",
    RADIO_VISUAL_KM,
    "km",
    flush=True
)


print(
    "Cargando modelo de densidad...",
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


print(
    "Cargando eventos historicos...",
    flush=True
)

eventos = pd.read_csv(
    EVENTS_PATH
)


# Limpiar los nombres de las columnas

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


# Convertir coordenadas a numeros

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


# Conservar solamente coordenadas validas

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


if len(eventos) == 0:

    raise ValueError(
        "eventos_historicos.csv no contiene "
        "coordenadas validas."
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


def obtener_indices_radio(
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


    return indices


def contar_eventos_radio(
    latitud,
    longitud,
    radio_km
):

    indices = obtener_indices_radio(
        latitud,
        longitud,
        radio_km
    )


    return int(
        len(indices)
    )


def obtener_conteos(
    latitud,
    longitud
):

    conteos = {

        "250m": contar_eventos_radio(
            latitud,
            longitud,
            0.25
        ),

        "500m": contar_eventos_radio(
            latitud,
            longitud,
            0.50
        ),

        "1km": contar_eventos_radio(
            latitud,
            longitud,
            1.00
        ),

        "2km": contar_eventos_radio(
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


def generar_imagen_densidad(
    latitud_centro,
    longitud_centro,
    radio_km=2.0
):

    print(
        "Generando imagen de densidad...",
        flush=True
    )


    # Buscar los eventos dentro del radio visual

    indices = obtener_indices_radio(
        latitud_centro,
        longitud_centro,
        radio_km
    )


    eventos_cercanos = coordenadas_eventos[
        indices
    ]


    print(
        "Eventos usados en la imagen:",
        len(eventos_cercanos),
        flush=True
    )


    # Calcular el rango geográfico de la imagen

    delta_latitud = (
        radio_km /
        111.0
    )


    coseno_latitud = np.cos(
        np.radians(
            latitud_centro
        )
    )


    # Protección para evitar división entre cero

    coseno_latitud = max(
        abs(coseno_latitud),
        0.000001
    )


    delta_longitud = (
        radio_km /
        (
            111.0 *
            coseno_latitud
        )
    )


    # Crear exactamente una imagen de 224 x 224

    figura, eje = plt.subplots(
        figsize=(
            2.24,
            2.24
        ),
        dpi=100
    )


    figura.patch.set_facecolor(
        "#f8fafc"
    )


    eje.set_facecolor(
        "#f8fafc"
    )


    # Dibujar los eventos históricos en rojo

    if len(
        eventos_cercanos
    ) > 0:

        eje.scatter(
            eventos_cercanos[:, 1],
            eventos_cercanos[:, 0],
            s=18,
            c="#dc2626",
            alpha=0.65,
            edgecolors="none"
        )


    # Dibujar la ubicación consultada con una X azul

    eje.scatter(
        longitud_centro,
        latitud_centro,
        s=55,
        c="#2563eb",
        marker="x",
        linewidths=2
    )


    # Mantener la misma escala de 2 km

    eje.set_xlim(
        longitud_centro - delta_longitud,
        longitud_centro + delta_longitud
    )


    eje.set_ylim(
        latitud_centro - delta_latitud,
        latitud_centro + delta_latitud
    )


    # Ocultar ejes, etiquetas y bordes

    eje.axis(
        "off"
    )


    plt.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0
    )


    # Guardar la imagen en memoria

    buffer = BytesIO()


    figura.savefig(
        buffer,
        format="png",
        dpi=100,
        pad_inches=0,
        facecolor="#f8fafc"
    )


    plt.close(
        figura
    )


    buffer.seek(0)


    # Abrir la imagen generada

    imagen = Image.open(
        buffer
    ).convert(
        "RGB"
    )


    # Garantizar 224 x 224

    if imagen.size != (
        224,
        224
    ):

        imagen = imagen.resize(
            (
                224,
                224
            ),
            Image.Resampling.BILINEAR
        )


    # Convertir a arreglo

    arreglo = np.asarray(
        imagen,
        dtype=np.float32
    )


    # Agregar dimensión batch:
    # (224,224,3) -> (1,224,224,3)

    entrada = np.expand_dims(
        arreglo,
        axis=0
    )


    buffer.close()


    print(
        "Forma de la imagen de densidad:",
        entrada.shape,
        flush=True
    )


    print(
        "Rango de pixeles:",
        float(
            entrada.min()
        ),
        "-",
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
            "Considerar monitoreo GPS, cambio de horario "
            "y medidas adicionales de seguridad."
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


    # 1. Contar eventos históricos

    conteos = obtener_conteos(
        latitud,
        longitud
    )


    # 2. Crear la imagen de densidad
    # Esta imagen reemplaza totalmente StaticMap

    entrada = generar_imagen_densidad(
        latitud,
        longitud,
        RADIO_VISUAL_KM
    )


    # 3. Ejecutar MobileNetV2

    print(
        "Ejecutando modelo de densidad...",
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


    # 4. Obtener clase ganadora

    indice = int(
        np.argmax(
            probabilidades
        )
    )


    clase = str(
        CLASES[
            indice
        ]
    )


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


    # 5. Probabilidades por clase

    desglose = {}


    for posicion, nombre_clase in enumerate(
        CLASES
    ):

        desglose[
            str(nombre_clase)
        ] = round(
            float(
                probabilidades[
                    posicion
                ]
            ) * 100,
            2
        )


    # 6. Resultado final

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
            ),

        "modelo_utilizado":
            MODEL_PATH,

        "tipo_imagen":
            "densidad_eventos",

        "radio_visual_km":
            RADIO_VISUAL_KM
    }


    print(
        "Resultado final:",
        resultado,
        flush=True
    )


    return resultado


@app.route(
    "/",
    methods=[
        "GET"
    ]
)
def inicio():

    return jsonify({

        "servicio":
            "API de evaluacion visual de riesgo por densidad",

        "estado":
            "activo",

        "modelo":
            MODEL_PATH,

        "clases":
            CLASES,

        "eventos_historicos_cargados":
            len(eventos),

        "tipo_imagen":
            "densidad_eventos",

        "radio_visual_km":
            RADIO_VISUAL_KM
    })


@app.route(
    "/salud",
    methods=[
        "GET"
    ]
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
    methods=[
        "GET"
    ]
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

        "tipo_imagen":
            "densidad_eventos",

        "radio_visual_km":
            RADIO_VISUAL_KM,

        "configuracion":
            CONFIGURACION,

        "radios_conteo": [
            "250m",
            "500m",
            "1km",
            "2km"
        ]
    })


@app.route(
    "/predecir",
    methods=[
        "POST"
    ]
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


        # Validar coordenadas

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
