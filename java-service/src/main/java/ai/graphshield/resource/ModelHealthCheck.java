package ai.graphshield.resource;

import org.eclipse.microprofile.health.HealthCheck;
import org.eclipse.microprofile.health.HealthCheckResponse;
import org.eclipse.microprofile.health.Readiness;

import ai.graphshield.service.GnnInferenceService;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;


@Readiness
@ApplicationScoped
public class ModelHealthCheck implements HealthCheck {
    
    @Inject
    GnnInferenceService inferenceService;
    
    @Override
    public HealthCheckResponse call() {
        boolean modelReady = inferenceService.isModelReady();
        
        return HealthCheckResponse.named("model-health")
                .status(modelReady)
                .withData("modelLoaded", modelReady)
                .build();
    }
}