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
