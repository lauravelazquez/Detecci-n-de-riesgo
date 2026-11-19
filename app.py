import os
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from staticmap import StaticMap, CircleMarker

app = Flask(__name__)
CORS(app)
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_riesgo.keras")
CLASES = ["ALTO", "BAJO", "MEDIO"]
UMBRAL = 0.65
modelo = tf.keras.models.load_model(MODEL_PATH)

preprocesamiento = tf.keras.Sequential([
    tf.keras.layers.CenterCrop(height=360, width=540),
    tf.keras.layers.Resizing(224, 224),
    tf.keras.layers.Rescaling(1./127.5, offset=-1)
])

def evaluar_ubicacion(latitud, longitud):
    mapa = StaticMap(600, 400)
    mapa.add_marker(CircleMarker((longitud, latitud), "#FF0000", 12))
    imagen = mapa.render(zoom=17)
    arreglo = tf.keras.utils.img_to_array(imagen)
    lote = tf.expand_dims(arreglo, 0)
    entrada = preprocesamiento(lote)
    pred = modelo.predict(entrada, verbose=0)[0]
    indice = int(np.argmax(pred))
    clase = CLASES[indice]
    confianza = float(pred[indice])
    if confianza < UMBRAL:
        estado = f"REVISIÓN MANUAL (Posible {clase})"
        protocolo = "El modelo no está suficientemente seguro. Validar la ruta con un analista."
    elif clase == "ALTO":
        estado = clase
        protocolo = "ALERTA: Asignar custodia o cambiar el horario de entrega."
    elif clase == "MEDIO":
        estado = clase
        protocolo = "PRECAUCIÓN: Mantener monitoreo GPS activo."
    else:
        estado = clase
        protocolo = "RUTA SEGURA: Operación normal."
    return {"coordenadas":[latitud,longitud],"riesgo_detectado":estado,"confianza_porcentaje":round(confianza*100,2),"desglose_probabilidades":{CLASES[i]:round(float(pred[i])*100,2) for i in range(len(CLASES))},"protocolo":protocolo}

@app.get("/")
def inicio(): return jsonify({"servicio":"API de evaluación de riesgo","estado":"activo"})

@app.get("/salud")
def salud(): return jsonify({"estado":"ok"})

@app.post("/predecir")
def predecir():
    try:
        datos=request.get_json(silent=True) or {}
        lat=float(datos.get("latitud")); lon=float(datos.get("longitud"))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return jsonify({"error":"Coordenadas fuera de rango."}),400
        return jsonify(evaluar_ubicacion(lat,lon))
    except (TypeError,ValueError): return jsonify({"error":"Latitud y longitud son obligatorias y deben ser numéricas."}),400
    except Exception as exc:
        app.logger.exception(exc)
        return jsonify({"error":"No fue posible procesar la ubicación."}),500

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT",5000)),debug=False)
