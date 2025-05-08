import os
import time
import logging
import numpy as np
import onnxruntime as ort
from typing import Tuple, Dict, List, Optional

from config import MODEL_PATH

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GNNInferenceService:
    """Servicio de inferencia para el modelo GNN exportado en formato ONNX."""
    
    def __init__(self, model_path: str = MODEL_PATH):
        """
        Inicializa la sesión de ONNX Runtime con el modelo.
        
        Args:
            model_path: Ruta al modelo ONNX
        """
        logger.info(f"Cargando modelo desde {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El modelo ONNX no existe en {model_path}")
            
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path, 
            providers=["CPUExecutionProvider"],
            sess_options=options
        )
    
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        logger.info(f"Modelo cargado exitosamente")
        logger.info(f"Entradas: {self.input_names}")
        logger.info(f"Salidas: {self.output_names}")
    
    def preprocess(self, features: List[float], edge_index: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesa los datos de entrada para la inferencia.
        
        Args:
            features: Vector de características de la transacción
            edge_index: Matriz de conexiones [2, num_edges]
            
        Returns:
            Tuple con tensores NumPy listos para inferencia
        """
        x = np.array(features, dtype=np.float32).reshape(1, -1)
        
        edge = np.array(edge_index, dtype=np.int64)
        
        return x, edge
    
    def predict(self, features: List[float], edge_index: List[List[int]]) -> float:
        """
        Realiza la inferencia en el modelo ONNX.
        
        Args:
            features: Vector de características del nodo
            edge_index: Índices de aristas en formato [2, num_edges]
            
        Returns:
            Probabilidad de fraude [0-1]
        """
        x, edge = self.preprocess(features, edge_index)
        
        inputs = {
            self.input_names[0]: x,
            self.input_names[1]: edge
        }
        
        start_time = time.time()
        
        outputs = self.session.run(None, inputs)
        
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Latencia de inferencia: {latency_ms:.2f} ms")
        
        probability = float(outputs[0].item())
        return probability
        
    def get_explanation(self, probability: float) -> Dict[str, float]:
        """
        Genera una explicación simplificada del resultado (demo).
        En un escenario real, se usaría LIME, SHAP o similar.
        
        Args:
            probability: Probabilidad de fraude
            
        Returns:
            Diccionario con factores de explicación
        """
        explanation = {
            "network_centrality": 0.3 * probability + 0.1,
            "transaction_amount": 0.2 * probability + 0.05,
            "temporal_pattern": 0.25 * probability + 0.2,
            "address_history": 0.25 * probability + 0.05
        }
        return explanation

gnn_service = GNNInferenceService()