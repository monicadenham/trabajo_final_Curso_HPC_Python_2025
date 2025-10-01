# 🔥 Trabajo final del curso ***Introducción a la programación HPC con Python y sus aplicaciones al campo de proceso de imágenes 2025***

## Análisis de cicatrices de incendios forestales en la Patagonia con PyCUDA

Mònica Denham, septiembre 2025


## 📁 Estructura
- `informe_final.ipynb`: Notebook con scripts, ejecuciones, gráficos, resultados y conclusiones.
- `calculos_incendios.py`: script completo con procesamiento secuencial y paralelo incluido en el informe.
- `data/`: imágenes satelitales usadas para el procesamiento propuesto.
- `images/`: imágenes usadas para el informe (PNG).
- `venv/`: entorno de ejecución para este repositorio. (`source venv/bin/activate` )



## ⚠️ Nota para evaluadores

Este notebook incluye celdas que utilizan **PyCUDA para aceleración en GPU**.  
Para ejecutarlas, se requiere:

- GPU NVIDIA con drivers y CUDA instalados.
- Entorno Python con PyCUDA, rasterio, etc.

**Los resultados mostrados en este notebook YA FUERON EJECUTADOS** en mi máquina local (NVIDIA GTX 1080Ti, CUDA 12.9).  
Pueden revisar los gráficos, tiempos y conclusiones sin necesidad de ejecutar el código.

🔒 Este repositorio es público porque contiene únicamente código y análisis reproducible. Los datos satelitales se descargan automáticamente desde fuentes públicas (Planetary Computer, Landsat, Sentinel-2). 


