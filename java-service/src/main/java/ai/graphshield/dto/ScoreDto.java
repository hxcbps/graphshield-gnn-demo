package ai.graphshield.dto;

import java.util.Map;


public class ScoreDto {
    
    private float probability;
    private String label;
    private Map<String, Float> explanation;
    
    public ScoreDto() {
        // Constructor por defecto requerido para serialización
    }
    
    public ScoreDto(float probability, String label) {
        this.probability = probability;
        this.label = label;
    }
    
    public ScoreDto(float probability, String label, Map<String, Float> explanation) {
        this.probability = probability;
        this.label = label;
        this.explanation = explanation;
    }
    
    public float getProbability() {
        return probability;
    }
    
    public void setProbability(float probability) {
        this.probability = probability;
        // Asegurar coherencia entre probabilidad y etiqueta
        if (probability > 0.5f && !"fraud".equals(label)) {
            this.label = "fraud";
        } else if (probability <= 0.5f && !"legit".equals(label)) {
            this.label = "legit";
        }
    }
    
    public String getLabel() {
        return label;
    }
    
    public void setLabel(String label) {
        this.label = label;
    }
    
    public Map<String, Float> getExplanation() {
        return explanation;
    }
    
    public void setExplanation(Map<String, Float> explanation) {
        this.explanation = explanation;
    }
}