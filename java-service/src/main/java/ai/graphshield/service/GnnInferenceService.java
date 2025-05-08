package ai.graphshield.service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.onnxruntime.engine.OrtEngine;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;


@ApplicationScoped
public class GnnInferenceService {

    private static final Logger LOG = Logger.getLogger(GnnInferenceService.class);

    @ConfigProperty(name = "graphshield.model.path", defaultValue = "/model/fraud_gnn.onnx")
    String modelPath;

    private ZooModel<NDList, NDList> model;
    private Predictor<NDList, NDList> predictor;
    private NDManager manager;


    @PostConstruct
    void initialize() {
        LOG.infof("Inicializando inferencia GNN – modelo: %s", modelPath);

        try {
            Path path = Paths.get(modelPath);
            if (!Files.exists(path)) {
                throw new IOException("El modelo ONNX no existe en " + modelPath);
            }

            manager = NDManager.newBaseManager();

            Criteria<NDList, NDList> criteria = Criteria.builder()
                    .setTypes(NDList.class, NDList.class)
                    .optModelPath(path)
                    .optTranslator(new PassthroughTranslator())
                    .optEngine(OrtEngine.ENGINE_NAME)   // "OnnxRuntime"
                    .optDevice(Device.cpu())
                    .build();

            model = criteria.loadModel();
            predictor = model.newPredictor();
            LOG.info("✅  Modelo ONNX cargado con éxito");

        } catch (IOException | ModelNotFoundException | MalformedModelException e) {
            throw new RuntimeException("Error al inicializar el servicio de inferencia", e);
        }
    }

  

    /**
     * @param features  lista de 165 valores (ya escalados) para 1 nodo
     * @param edgeIndex lista [2, num_edges] (IDs relativos a este batch)
     */
    public float predict(List<Float> features, List<List<Integer>> edgeIndex) {

        if (!isModelReady()) {
            throw new IllegalStateException("Modelo no cargado");
        }
        if (features == null || features.size() != 165) {
            throw new IllegalArgumentException("Se requieren 165 features escaladas");
        }
        if (edgeIndex == null || edgeIndex.size() != 2
                || edgeIndex.get(0).size() != edgeIndex.get(1).size()) {
            throw new IllegalArgumentException("edge_index debe ser [2, n] y de igual longitud");
        }

        float[] f = new float[165];
        for (int i = 0; i < 165; i++) f[i] = features.get(i);
        NDArray x = manager.create(f, new Shape(1, 165));

        int e = edgeIndex.get(0).size();
        long[][] edges = new long[2][e];
        for (int i = 0; i < e; i++) {
            edges[0][i] = edgeIndex.get(0).get(i);
            edges[1][i] = edgeIndex.get(1).get(i);
        }
        NDArray edgeArr = manager.create(edges);

        try {
            NDList out = predictor.predict(new NDList(x, edgeArr));
            return out.get(0).toFloatArray()[0];
        } catch (Exception ex) {
            throw new RuntimeException("Inferencia ONNX falló", ex);
        }
    }

    public boolean isModelReady() {
        return model != null && predictor != null;
    }

    public Map<String, Float> getExplanation(float p) {
        Map<String, Float> e = new HashMap<>();
        e.put("network_centrality", 0.3f * p + 0.1f);
        e.put("transaction_amount", 0.2f * p + 0.05f);
        e.put("temporal_pattern",  0.25f * p + 0.2f);
        e.put("address_history",   0.25f * p + 0.05f);
        return e;
    }

    private static final class PassthroughTranslator implements Translator<NDList, NDList> {
        public NDList processInput(TranslatorContext c, NDList in) { return in; }
        public NDList processOutput(TranslatorContext c, NDList out) { return out; }
        public Batchifier getBatchifier() { return null; }
    }
}
