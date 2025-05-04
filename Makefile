.PHONY: setup train build up bench
setup:
\tconda env create -f training/environment.yml || true
\t@echo "⚡ Activa con  'conda activate gnn'"
\t@echo "👉  Descarga dataset  'dvc pull'  (cuando el .dvc esté añadido)"
