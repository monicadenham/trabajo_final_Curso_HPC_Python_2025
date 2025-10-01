
import numpy as np
import time
import rasterio
import pycuda.driver as drv
import matplotlib.pyplot as plt
import os
import pycuda.autoinit
from   pycuda.compiler import SourceModule


# constante que se usa para aumentar el cómputo
# al iterar el procesamiento varias veces
ITERACIONES = 10

print("✅ Librerías importadas 🔥.")


# Funcion para cargar una banda de un archivo raster usando rasterio
def cargar_banda(ruta):
    with rasterio.open(ruta) as src:
        return src.read(1).astype('float32')  # Lee la primera banda como float32



# Se cargan las bandas NIR, SWIR2  y RED del pre incendio y post incendio
# estas son las bandas necesarias para calcular el NDVI y el dNBR 
nir_pre = cargar_banda("../data/pre_fire_2021_SR_B5.tif")
swir2_pre = cargar_banda("../data/pre_fire_2021_SR_B7.tif")
nir_post = cargar_banda("../data/post_fire_2022_SR_B5.tif")
swir2_post = cargar_banda("../data/post_fire_2022_SR_B7.tif")
red_pre = cargar_banda("../data/pre_fire_2021_SR_B4.tif")
red_post = cargar_banda("../data/post_fire_2022_SR_B4.tif")

print("✅ Bandas cargadas.")
print(f"✅ Shape imágenes: {nir_pre.shape} 📐")

# Verificación de valores máximos y mínimos de las bandas cargadas. Esto se hace ya que hubo problemas con escaleo de las imágenes previamente. 
print("✅ NIR pre min/max:", nir_pre.min(), nir_pre.max(), "NIR post min/max: ", nir_post.min(), nir_post.max())
print("✅ SWIR2 pre min/max:", swir2_pre.min(), swir2_pre.max(), "SWIR2 post min/max:", swir2_post.min(), swir2_post.max())
print("✅ RED pre min/max:", red_pre.min(), red_pre.max(), "RED post min/max:", red_post.min(), red_post.max(),)



# Se calcula el NDVI pre y post incendio en CPU
# el cálculo es NBR = (NIR - SWIR2) / (NIR + SWIR2 + 1e-8)
filas = nir_pre.shape[0]
cols = nir_pre.shape[1]
ndvi_pre = np.zeros((filas, cols), dtype=np.float32)
ndvi_post = np.zeros((filas, cols), dtype=np.float32)

# cronometro inicio
sec_start = time.time()
for i in range(ITERACIONES):  # Aumento el cómputo iterando el procesamiento
    # loop principal
    for f in range(filas):
        for c in range(cols):
            # ndvi pre incendio
            ndvi_pre[f,c] = (nir_pre[f,c] - red_pre[f,c]) / (nir_pre[f,c] + red_pre[f,c] + 1e-8)
            # nbr post incendio
            ndvi_post[f,c] = (nir_post[f,c] - red_post[f,c]) / (nir_post[f,c] + red_post[f,c] + 1e-8)
#cronometro final
sec_end = time.time()

# resultados
print(f"✋✋✋✋✋ Resultados secuenciales ✋✋✋✋✋")
print(f"✅ Cálculo secuencial del NDVI pre incendio finalizado.")
print(f"✅ Cálculo secuencial del NDVI post incendio finalizado.")
print(f"✅ 🕜 Cálculo secuencial del NDVI finalizado en {sec_end - sec_start:.4f} segundos.")

# almaceno tiempos para análisis de rendimiento
cpu_ndvi_t = sec_end - sec_start


# Se calcula el NBR pre y post incendio en CPU
# el cálculo es NBR = (NIR - SWIR2) / (NIR + SWIR2 + 1e-8)
filas = nir_pre.shape[0]
cols = nir_pre.shape[1]
dnbr_sec = np.zeros((filas, cols), dtype=np.float32)

sec_start = time.time()
for i in range(ITERACIONES):  # Aumento el cómputo iterando el procesamiento
    for f in range(filas):
        for c in range(cols):
            # nbr pre incendio
            nbr_pre = (nir_pre[f,c] - swir2_pre[f,c]) / (nir_pre[f,c] + swir2_pre[f,c] + 1e-8)
            # nbr post incendio
            nbr_post = (nir_post[f,c] - swir2_post[f,c]) / (nir_post[f,c] + swir2_post[f,c] + 1e-8)

            d_nbr = nbr_pre - nbr_post
            # guardo el resultado en un raster de salida  
            dnbr_sec[f,c] = d_nbr
sec_end = time.time()

# resultados
print(f"✋✋✋✋✋ Resultados secuenciales ✋✋✋✋✋")
print(f"✅ 🕜 Cálculo secuencial del dNBR finalizado en {sec_end - sec_start:.4f} segundos.")

# almaceno tiempos para análisis de rendimiento
cpu_dnbr_t = sec_end - sec_start



# Reserva memoria en GPU, datos de entrada pre incendio
d_nir_pre = drv.mem_alloc(nir_pre.nbytes)
d_swir2_pre = drv.mem_alloc(swir2_pre.nbytes)
d_red_pre = drv.mem_alloc(red_pre.nbytes)
# Reserva memoria en GPU, datos de entrada post incendio
d_nir_post = drv.mem_alloc(nir_post.nbytes)
d_swir2_post = drv.mem_alloc(swir2_post.nbytes)
d_red_post = drv.mem_alloc(red_post.nbytes)

# Reserva de memoria en GPU para resultados
d_ndvi_pre = drv.mem_alloc(ndvi_pre.nbytes)
d_ndvi_post = drv.mem_alloc(ndvi_post.nbytes)
d_dnbr_final = drv.mem_alloc(dnbr_sec.nbytes)


# Transferencia de datos host->GPU de datos de entrada
drv.memcpy_htod(d_nir_pre, nir_pre)
drv.memcpy_htod(d_swir2_pre, swir2_pre)
drv.memcpy_htod(d_red_pre, red_pre)
drv.memcpy_htod(d_nir_post, nir_post)
drv.memcpy_htod(d_swir2_post, swir2_post)
drv.memcpy_htod(d_red_post, red_post)

print("✅ Memoria reservada en GPU.")
print("✅ Datos transferidos a GPU.")


# Armado de la grilla 2D con bloques 2D para mapear threads con imagen
bloque = (32, 32, 1)
grilla = ( (cols + bloque[0] - 1) // bloque[0], (filas + bloque[1] - 1) // bloque[1] )
print("📐 Dimensión de cada bloque (x,y,z): ", bloque)    
print("📐 Dimensión de la grilla: (x,y,z)", grilla)

# Declaracion de timers para GPU
start_evt = drv.Event()
end_evt = drv.Event()
print("✅ Clocks creados. 🕜")

# Definición del kernel
mod = SourceModule ("""
__global__ void calcular_NDVI_paralelo(float *d_nir, float *d_red, float *d_ndvi, int filas, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * cols + col;
    
                        
    if (index < filas * cols) {
       d_ndvi[index] = (d_nir[index] - d_red[index]) / (d_nir[index] + d_red[index] + 1e-8);
     
    }
}
""")
elapsed_time_par = 0.0
# Ejecución del kernel y cronometraje. Se invoca al kernel dos veces, para calcular en NDVI pre y post incendio
par_start = time.time()
for i in range(ITERACIONES):  # Aumento el cómputo iterando el procesamiento
    start_evt.record()  # Inicio del timer GPU
    calcular_ndvi_paralelo = mod.get_function("calcular_NDVI_paralelo")
     # calculo del ndvi pre incendio
    calcular_ndvi_paralelo(d_nir_pre, d_red_pre, d_ndvi_pre, np.int32(filas), np.int32(cols), block=bloque, grid=grilla)
    # calculo del ndvi post incendio
    calcular_ndvi_paralelo(d_nir_post, d_red_post, d_ndvi_post, np.int32(filas), np.int32(cols), block=bloque, grid=grilla)
    end_evt.record()  # Fin del timer GPU
    end_evt.synchronize()  # Espera a que el evento de fin se complete  
    elapsed_time_par += start_evt.time_till(end_evt)  # Tiempo transcurrido en milisegundos    
par_end = time.time()

# Transferir datos GPU->host
ndvi_pre_par = np.empty_like(ndvi_pre)
ndvi_post_par = np.empty_like(ndvi_post)
drv.memcpy_dtoh(ndvi_pre_par, d_ndvi_pre)
drv.memcpy_dtoh(ndvi_post_par, d_ndvi_post )

# resultados
print(f"✋✋✋✋✋ Resultados PARALELOS ✋✋✋✋✋")
print("✅ Cálculo del NVDI previo al incendio en paralelo.")
print("✅ Cálculo del NVDI posterior al incendio en paralelo.")
print("✅ 🕜 Paralelo. Tiempo cálculo: ", par_end - par_start, "segundos.")
print("✅ 🕜 Paralelo. Tiempo cálculo usando eventos: ", elapsed_time_par *1e-3, "segundos.")

# almaceno tiempos para análisis de rendimiento
gpu_ndvi_t = par_end - par_start


# Definición del kernel
mod = SourceModule ("""
__global__ void calcular_dNBR_paralelo(float *d_nir_pre, float *d_swir2_pre, float *d_nir_post, float *d_swir2_post, float *d_dnbr_final, int filas, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * cols + col;
    float nbr_pre, nbr_post, d_nbr;
                        
    if (index < filas * cols) {
        nbr_pre = (d_nir_pre[index] - d_swir2_pre[index]) / (d_nir_pre[index] + d_swir2_pre[index] + 1e-8);
        nbr_post = (d_nir_post[index] - d_swir2_post[index]) / (d_nir_post[index] + d_swir2_post[index] + 1e-8);
        d_nbr = nbr_pre - nbr_post;                
        d_dnbr_final[index] = d_nbr;
    }
}
""")

# Ejecución del kernel y cronometraje
par_start = time.time()
for i in range(ITERACIONES):  # Aumento el cómputo iterando el procesamiento
    calcular_dnbr_paralelo = mod.get_function("calcular_dNBR_paralelo")
    calcular_dnbr_paralelo(d_nir_pre, d_swir2_pre, d_nir_post, d_swir2_post, d_dnbr_final, np.int32(filas), np.int32(cols), block=bloque, grid=grilla)

par_end = time.time()

# Transferir datos GPU->host
dnbr_final_par = np.empty_like(dnbr_sec)
drv.memcpy_dtoh(dnbr_final_par, d_dnbr_final)

# almaceno tiempos para análisis de rendimiento
gpu_dnbr_t = par_end - par_start
print("✅ 🕜 Paralelo. Tiempo cálculo: ", par_end - par_start, "segundos.")

