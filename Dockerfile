FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

COPY app.py .

COPY modelo_riesgo_densidad.keras .
COPY clases_riesgo_densidad.json .
COPY configuracion_riesgo_densidad.json .
COPY eventos_historicos.csv .

ENV PORT=10000

EXPOSE 10000

CMD [
    "gunicorn",
    "--workers", "1",
    "--threads", "1",
    "--timeout", "180",
    "--bind", "0.0.0.0:10000",
    "app:app"
]
