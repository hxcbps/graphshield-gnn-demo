from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field, model_validator

class Tx(BaseModel):
    """
    Representación de una transacción para inferencia.
    
    Campos:
    - features: Vector de características de la transacción
    - edge_index: Matriz de índices de conexiones [2, E]
    """
    features: List[float] = Field(..., description="Vector de características de la transacción")
    edge_index: List[List[int]] = Field(..., description="Aristas en formato [2, num_edges]")
    
    @model_validator(mode='after')
    def validate_edge_index(self):
        """Valida que edge_index tenga el formato correcto [2, num_edges]"""
        if len(self.edge_index) != 2:
            raise ValueError("edge_index debe ser una lista de 2 elementos [src_indices, dst_indices]")
        return self

class Score(BaseModel):
    """
    Resultado de la clasificación de fraude.
    
    Campos:
    - probability: Probabilidad de fraude [0-1]
    - label: Etiqueta ("fraud" o "legit")
    """
    probability: float = Field(..., description="Probabilidad de fraude [0-1]", ge=0.0, le=1.0)
    label: str = Field(..., description="Etiqueta de clasificación")
    explanation: Optional[Dict[str, float]] = Field(None, description="Factores que contribuyen a la decisión (si se solicita)")
    
    @model_validator(mode='after')
    def validate_label(self):
        """Asegura coherencia entre probabilidad y etiqueta"""
        if self.probability > 0.5 and self.label != "fraud":
            self.label = "fraud"
        elif self.probability <= 0.5 and self.label != "legit":
            self.label = "legit"
        return self