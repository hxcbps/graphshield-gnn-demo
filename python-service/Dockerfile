FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY *.py .

# Crear directorio para el modelo
RUN mkdir -p /model

# Variables de entorno
ENV MODEL_DIR=/model
ENV SERVICE_PORT=9443
ENV SERVICE_HOST=0.0.0.0

# Puerto
EXPOSE 9443

# Comando por defecto
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9443"]