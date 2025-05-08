import time
import logging
from typing import Optional

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from schemas import Tx, Score
from gnn_service import gnn_service
from config import SERVICE_PORT

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GraphShield-GNN Python",
    description="Microservicio de detección de fraude en tiempo real con Graph Neural Networks",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,  
    excluded_handlers=["/docs", "/openapi.json"],
)

instrumentator.instrument(app).expose(app)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
def read_root():
    """Endpoint principal para health check."""
    return {"status": "healthy", "service": "GraphShield-GNN Python"}

@app.get("/health")
def health_check():
    """Endpoint de health check."""
    try:
        # Verificar que el modelo está cargado correctamente
        if gnn_service.session is None:
            return {"status": "error", "message": "Modelo no disponible"}
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/score", response_model=Score)
def score_transaction(tx: Tx, explain: Optional[bool] = False):
    """
    Endpoint para clasificar una transacción como fraudulenta o legítima.
    
    Args:
        tx: Datos de la transacción (features + grafo)
        explain: Si se debe generar una explicación
        
    Returns:
        Puntuación de riesgo y etiqueta
    """
    try:
        logger.debug(f"Recibida petición de scoring")
        probability = gnn_service.predict(tx.features, tx.edge_index)
        label = "fraud" if probability > 0.5 else "legit"
        explanation = None
        if explain:
            explanation = gnn_service.get_explanation(probability)
        
        return Score(
            probability=probability,
            label=label,
            explanation=explanation
        )
    except ValueError as ve:
        logger.error(f"Error de validación: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error durante scoring: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al procesar la petición")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Iniciando servicio en puerto {SERVICE_PORT}")
    uvicorn.run("main:app", host="0.0.0.0", port=SERVICE_PORT, reload=False)