import os
from pathlib import Path

# Ruta al modelo ONNX
MODEL_DIR = os.environ.get("MODEL_DIR", "../model")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_gnn.onnx")

# Servicio
SERVICE_PORT = int(os.environ.get("SERVICE_PORT", 9443))
SERVICE_HOST = os.environ.get("SERVICE_HOST", "0.0.0.0")

# Métricas
METRICS_PATH = "/metrics"

# Kafka (si se usa para ingesta)
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "tx-json")