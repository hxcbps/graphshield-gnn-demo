package ai.graphshield.resource;

import org.jboss.logging.Logger;

import ai.graphshield.dto.ScoreDto;
import ai.graphshield.dto.TxDto;
import ai.graphshield.service.GnnInferenceService;
import ai.graphshield.service.MappingService;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.DefaultValue;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.QueryParam;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import java.util.Map;
import java.util.HashMap;

/**
 * Recurso REST para el servicio de scoring.
 */
@Path("/")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
public class ScoringResource {
    
    private static final Logger LOG = Logger.getLogger(ScoringResource.class);
    
    @Inject
    GnnInferenceService inferenceService;
    
    @Inject
    MappingService mappingService;
    

    @GET
    public Response root() {
        Map<String, String> response = new HashMap<>();
        response.put("status", "healthy");
        response.put("service", "GraphShield-GNN Java");
        return Response.ok(response).build();
    }
    
    /**
     * Endpoint para clasificar una transacción como fraudulenta o legítima.
     * 
     * @param txDto Datos de la transacción
     * @param explain Si se debe generar una explicación
     * @return Puntuación de riesgo y etiqueta
     */
    @POST
    @Path("/score")
    public ScoreDto scoreTransaction(TxDto txDto, 
                                   @QueryParam("explain") @DefaultValue("false") boolean explain) {
        try {
            LOG.debug("Recibida petición de scoring");
            
            mappingService.validateTxDto(txDto);
            
            float probability = inferenceService.predict(txDto.getFeatures(), txDto.getEdgeIndex());

            Map<String, Float> explanation = null;
            if (explain) {
                explanation = inferenceService.getExplanation(probability);
            }
            
            return mappingService.createScoreDto(probability, explain, explanation);
            
        } catch (IllegalArgumentException e) {
            LOG.error("Error de validación: " + e.getMessage());
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("error", e.getMessage());
            throw new WebApplicationException(
                    Response.status(Response.Status.BAD_REQUEST)
                            .entity(errorResponse)
                            .build());
                            
        } catch (Exception e) {
            LOG.error("Error durante scoring: " + e.getMessage(), e);
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("error", "Error interno al procesar la petición");
            throw new WebApplicationException(
                    Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                            .entity(errorResponse)
                            .build());
        }
    }
}