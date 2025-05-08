package ai.graphshield.dto;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;


public class TxDto {
    
    private List<Float> features;
    private List<List<Integer>> edgeIndex;
    
    public TxDto() {
        
    }
    
    public List<Float> getFeatures() {
        return features;
    }
    
    public void setFeatures(List<Float> features) {
        this.features = features;
    }
    
    @JsonProperty("edge_index")
    public List<List<Integer>> getEdgeIndex() {
        return edgeIndex;
    }
    
    public void setEdgeIndex(List<List<Integer>> edgeIndex) {
        this.edgeIndex = edgeIndex;
    }
    
    /**
     * Valida que la estructura de edgeIndex sea correcta.
     * 
     * @return true si la estructura es válida
     * @throws IllegalArgumentException si la estructura no es válida
     */
    public boolean validateEdgeIndex() {
        if (edgeIndex == null || edgeIndex.size() != 2) {
            throw new IllegalArgumentException("edge_index debe ser una lista de 2 elementos [src_indices, dst_indices]");
        }
        
        // Verificar que ambas listas tengan el mismo tamaño
        if (edgeIndex.get(0).size() != edgeIndex.get(1).size()) {
            throw new IllegalArgumentException("Las listas de src_indices y dst_indices deben tener el mismo tamaño");
        }
        
        return true;
    }
}