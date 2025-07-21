# Analítica de Clientes – Proceso de Selección

Este repositorio contiene el desarrollo de un caso práctico de analítica de datos aplicado a un conjunto de transacciones de clientes. Incluye desde el procesamiento inicial hasta la segmentación mediante clustering.

---

## Estructura del proyecto

```text
├── data/              # Datos en tres niveles: raw, procesado y listo para modelar
│   ├── bronze/        # Datos crudos originales (Excel entregado)
│   ├── silver/        # Datos limpios y consolidados (output de data_silver.py)
│   └── gold/          # Datos con features listas para el modelo (output de data_gold.py)
│
├── docs/              # Documentación del caso entregada por la empresa
│
├── notebooks/         # Exploración y modelado en Jupyter (EDA, segmentación, reglas)
│
├── results/           # Resultados del modelo entrenado y visualizaciones
│   ├── data/          # Datos con los clusters asignados
│   ├── models/        # Modelo K-Means entrenado (.pkl)
│   └── pics/          # Gráficos generados durante el análisis y visualización
│
├── scripts/           # Scripts Python que ejecutan cada etapa del pipeline
│   ├── data_silver.py # Limpieza y consolidación de datos
│   ├── data_gold.py   # Agregación y feature engineering
│   ├── model.py       # Entrenamiento del modelo de clustering
│   └── utils.py       # Funciones auxiliares reutilizables
│
├── main.py            # Script principal (Ideal para orquestacion, app de Streamlit)
├── requirements.txt   # Dependencias del proyecto
├── .gitignore         # Exclusiones para Git
```
---
# Notas adicionales

 * El proyecto puede escalar a clustering jerárquico o modelos supervisados si se incorporan etiquetas.
 * Las visualizaciones clave están disponibles en results/pics/.
 * La asignación de cluster por cliente se encuentra en results/data/clustered_data.csv.


