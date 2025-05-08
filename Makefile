.PHONY: setup train build up down restart logs bench help

# Variables
PYTHON_SERVICE_DIR = python-service
JAVA_SERVICE_DIR = java-service
CONDA_ENV = gnn

help:
	@echo "GraphShield-GNN Demo Makefile"
	@echo "----------------------------"
	@echo "Comandos disponibles:"
	@echo "  setup   : Configura el entorno conda y descarga datasets con DVC"
	@echo "  train   : Entrena el modelo GNN y lo exporta a ONNX"
	@echo "  build   : Construye las imágenes Docker de los servicios"
	@echo "  up      : Levanta todos los servicios con docker-compose"
	@echo "  down    : Detiene todos los servicios"
	@echo "  restart : Reinicia todos los servicios"
	@echo "  logs    : Muestra logs de los servicios"
	@echo "  bench   : Ejecuta benchmark de comparación entre servicios"

setup:
	@echo "Configurando entorno..."
	conda env create -f training/environment.yml
	dvc pull

train:
	@echo "Activando entorno conda y entrenando modelo..."
	bash training/run_training_pipeline.sh

build:
	@echo "Construyendo imágenes Docker..."
	docker-compose -f infrastructure/docker-compose.yml build

up:
	@echo "Levantando servicios..."
	docker-compose -f infrastructure/docker-compose.yml up -d
	@echo "Servicios disponibles en:"
	@echo "- Python: http://localhost:9443"
	@echo "- Java: http://localhost:9444"
	@echo "- Grafana: http://localhost:3000 (admin/graphshield)"
	@echo "- Prometheus: http://localhost:9090"
	@echo "- Jaeger: http://localhost:16686"

down:
	@echo "Deteniendo servicios..."
	docker-compose -f infrastructure/docker-compose.yml down

restart:
	@echo "Reiniciando servicios..."
	docker-compose -f infrastructure/docker-compose.yml restart

logs:
	@echo "Mostrando logs..."
	docker-compose -f infrastructure/docker-compose.yml logs -f

bench:
	@echo "Ejecutando benchmark..."
	bash benchmark/run_load_test.sh