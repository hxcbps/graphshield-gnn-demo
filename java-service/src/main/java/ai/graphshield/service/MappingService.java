package ai.graphshield.service;

import java.util.HashMap;
import java.util.Map;          
import java.util.List;          

import ai.graphshield.dto.ScoreDto;
import ai.graphshield.dto.TxDto;
import jakarta.enterprise.context.ApplicationScoped;

/**
 * Servicio para validación y mapeo de DTOs.
 */
@ApplicationScoped
public class MappingService {

    public void validateTxDto(TxDto dto) {
        if (dto == null)
            throw new IllegalArgumentException("El DTO no puede ser nulo");

        List<Float> feats = dto.getFeatures();
        if (feats == null || feats.size() != 165)
            throw new IllegalArgumentException("Se requieren 165 features");

        dto.validateEdgeIndex();
    }

    public ScoreDto createScoreDto(float p, boolean explain, Map<String, Float> exp) {
        String label = p > 0.5f ? "fraud" : "legit";
        return explain ? new ScoreDto(p, label, exp)
                       : new ScoreDto(p, label);
    }
}
