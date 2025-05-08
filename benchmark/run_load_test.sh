#!/bin/bash
# Script para ejecutar pruebas de carga contra ambos servicios

# Definir colores para la salida
GREEN="\033[0;32m"
BLUE="\033[0;34m"
RED="\033[0;31m"
NC="\033[0m" # No Color

echo -e "${BLUE}=== GraphShield-GNN Load Test ===${NC}"
echo "Este script ejecutará pruebas de carga contra ambos servicios"

# Verificar que wrk esté instalado
if ! command -v wrk &> /dev/null; then
    echo -e "${RED}Error: 'wrk' no está instalado. Por favor instálalo antes de continuar.${NC}"
    echo "Ubuntu/Debian: sudo apt-get install -y wrk"
    echo "macOS: brew install wrk"
    exit 1
fi

# Verificar que jq esté instalado
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: 'jq' no está instalado. Por favor instálalo antes de continuar.${NC}"
    echo "Ubuntu/Debian: sudo apt-get install -y jq"
    echo "macOS: brew install jq"
    exit 1
fi

# Crear directorio para resultados
RESULTS_DIR="benchmark/results"
mkdir -p $RESULTS_DIR

# Generar JSON de ejemplo para la petición
echo -e "${BLUE}Generando datos de prueba...${NC}"
cat > $RESULTS_DIR/sample_tx.json << EOF
{
  "features": [$(for i in $(seq 1 165); do printf "0.%03d" "$i"; [ $i -lt 165 ] && printf ", "; done)],
  "edge_index": [[0,1,2,3],[1,2,3,0]]
}
EOF

# Crear script LUA para wrk
cat > benchmark/post_json.lua << EOF
-- Leer el contenido del archivo JSON
function read_file(path)
    local file = io.open(path, "r")
    if not file then return nil end
    local content = file:read("*a")
    file:close()
    return content
end

-- Obtener el payload desde el archivo
local payload = read_file("benchmark/results/sample_tx.json")

-- Configurar la petición
request = function()
   return wrk.format("POST", "/score", {
      ["Content-Type"] = "application/json",
      ["Accept"] = "application/json"
   }, payload)
end
EOF

# Función para ejecutar la prueba
run_benchmark() {
    local service=$1
    local url=$2
    local duration=$3
    local connections=$4
    local threads=$5
    
    echo -e "${GREEN}Ejecutando prueba para $service${NC}"
    echo "URL: $url"
    echo "Duración: $duration, Conexiones: $connections, Threads: $threads"
    
    # Ejecutar wrk
    wrk -t$threads -c$connections -d$duration -s benchmark/post_json.lua \
        --latency $url > $RESULTS_DIR/${service}_results.txt
    
    # Mostrar resultados
    cat $RESULTS_DIR/${service}_results.txt
    echo -e "${GREEN}Resultados guardados en $RESULTS_DIR/${service}_results.txt${NC}"
    echo ""
}

# Parámetros de la prueba
DURATION=${1:-30s}  
CONNECTIONS=${2:-50}  
THREADS=${3:-4}  

# Ejecutar pruebas para ambos servicios
echo -e "${BLUE}Iniciando pruebas de carga...${NC}"

# Prueba Python
run_benchmark "python" "http://localhost:9443/score" $DURATION $CONNECTIONS $THREADS

# Prueba Java
run_benchmark "java" "http://localhost:9444/score" $DURATION $CONNECTIONS $THREADS

echo -e "${BLUE}=== Pruebas completadas ===${NC}"


extract_metric() {
    local file=$1
    local metric=$2
    local value=$(grep "$metric" $file | awk '{print $2}')
    echo $value
}

PYTHON_RESULTS="$RESULTS_DIR/python_results.txt"
JAVA_RESULTS="$RESULTS_DIR/java_results.txt"

if [[ -f $PYTHON_RESULTS && -f $JAVA_RESULTS ]]; then
    echo -e "${BLUE}=== Comparación de Resultados ===${NC}"
    
    # Requests/sec
    PYTHON_RPS=$(grep "Requests/sec" $PYTHON_RESULTS | awk '{print $2}')
    JAVA_RPS=$(grep "Requests/sec" $JAVA_RESULTS | awk '{print $2}')
    
    # Latencias
    PYTHON_AVG=$(grep -A 3 "Latency" $PYTHON_RESULTS | grep "Avg" | awk '{print $2}')
    JAVA_AVG=$(grep -A 3 "Latency" $JAVA_RESULTS | grep "Avg" | awk '{print $2}')
    
    PYTHON_P99=$(grep -A 3 "Latency" $PYTHON_RESULTS | grep "99%" | awk '{print $2}')
    JAVA_P99=$(grep -A 3 "Latency" $JAVA_RESULTS | grep "99%" | awk '{print $2}')
    
    echo -e "                   ${GREEN}Python${NC}    ${GREEN}Java${NC}"
    echo -e "Requests/sec:      ${GREEN}$PYTHON_RPS${NC}    ${GREEN}$JAVA_RPS${NC}"
    echo -e "Latencia media:    ${GREEN}$PYTHON_AVG${NC}   ${GREEN}$JAVA_AVG${NC}"
    echo -e "Latencia P99:      ${GREEN}$PYTHON_P99${NC}   ${GREEN}$JAVA_P99${NC}"
    
    echo ""
    echo -e "${BLUE}Nota:${NC} Revisa los archivos de resultados para análisis detallado."
fi

echo -e "${BLUE}=== Fin del benchmark ===${NC}"