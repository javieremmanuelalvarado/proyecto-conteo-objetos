# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 22:34:10 2026

@author: emman
"""
import numpy as np
import matplotlib.pyplot as plt

from contador_objetos.preprocesamiento import (
    leer_imagen,
    convertirBGR_a_grises,
    bgr_a_rgb,
    aplicar_convolucion,
    filtro_promedio,
    kernel_sobel_x,
    kernel_sobel_y,
    calcular_magnitud_gradiente,
    transformacion_negativa,
    calcular_matriz_harris,
    detectar_esquinas_harris
)

from contador_objetos.segmentacion import (
    umbralizar_imagen,
    calcular_umbral_otsu,
    etiquetar_componentes_conectados,
    eliminar_objetos_pequenos,
    calcular_propiedades
)



# CONFIGURACIÓN GENERAL

ruta = "imagenes_prueba/ejemplo_1.jpeg"

#Umbral automatico con otsu
#umbral_manual = 65

min_area = 5700           # Área mínima para conservar objetos
umbral_harris = 0.05    # Sensibilidad para detección de esquinas



# LECTURA Y PREPROCESAMIENTO

# Se carga la imagen original desde la carpeta de imágenes de prueba.
imagen = leer_imagen(ruta)

# OpenCV carga las imágenes en BGR, por eso se convierte a RGB solo para visualizarla bien.
imagen_rgb = bgr_a_rgb(imagen)

# La imagen se convierte a escala de grises para simplificar el procesamiento.
imagen_gris = convertirBGR_a_grises(imagen)

# Se aplica una transformación negativa como ejemplo de transformación de intensidad.
imagen_negativa = transformacion_negativa(imagen_gris)

# Se suaviza la imagen con un filtro promedio aplicado mediante convolución manual.
kernel_promedio = filtro_promedio()
imagen_suavizada = aplicar_convolucion(imagen_gris, kernel_promedio)



# DETECCIÓN DE BORDES CON SOBEL

# Se obtienen los kernels de Sobel para calcular cambios en X y en Y.
sobel_x = kernel_sobel_x()
sobel_y = kernel_sobel_y()

# Se calculan los gradientes sin recortar valores, porque Sobel puede generar valores negativos.
gradiente_x = aplicar_convolucion(imagen_suavizada, sobel_x, aplicar_clip=False)
gradiente_y = aplicar_convolucion(imagen_suavizada, sobel_y, aplicar_clip=False)

# Se combinan ambos gradientes para obtener la magnitud total del borde.
bordes = calcular_magnitud_gradiente(gradiente_x, gradiente_y)



# UMBRALIZACIÓN

# Se puede usar Otsu para calcular un umbral automático o un valor fijo manual.
umbral = calcular_umbral_otsu(imagen_suavizada)
print("Umbral automático Otsu:", umbral)

imagen_binaria = umbralizar_imagen(imagen_suavizada, umbral)

# Si en alguna imagen los objetos quedan negros y el fondo blanco, se puede invertir:
imagen_binaria = 255 - imagen_binaria


# SEGMENTACIÓN Y CONTEO

# Se identifican regiones conectadas dentro de la imagen binaria.
etiquetas, numero_objetos = etiquetar_componentes_conectados(imagen_binaria)

# Se eliminan regiones demasiado pequeñas para reducir ruido.
etiquetas_filtradas, numero_objetos_filtrados = eliminar_objetos_pequenos(
    etiquetas,
    min_area
)

# Se calculan propiedades geométricas básicas de cada objeto detectado.
propiedades = calcular_propiedades(etiquetas_filtradas)



# DETECCIÓN DE ESQUINAS CON HARRIS

# Harris se calcula sobre la imagen suavizada para reducir falsas detecciones por ruido.
respuesta_harris = calcular_matriz_harris(imagen_suavizada, k=0.04)

# Se detectan las esquinas usando un umbral relativo.
esquinas = detectar_esquinas_harris(
    respuesta_harris,
    umbral_relativo=umbral_harris
)

# Para visualizar Harris correctamente, se normaliza la respuesta al rango 0-255.
harris_visual = respuesta_harris - np.min(respuesta_harris)

if np.max(harris_visual) != 0:
    harris_visual = harris_visual / np.max(harris_visual)

harris_visual = (harris_visual * 255).astype(np.uint8)



# RESULTADOS EN CONSOLA

print("Número de objetos detectados antes del filtrado:", numero_objetos)
print("Número de objetos detectados después del filtrado:", numero_objetos_filtrados)
print("Número de esquinas detectadas:", len(esquinas))
print()

for obj in propiedades:
    print(f"Objeto {obj['etiqueta']}:")
    print(f"  Área = {obj['area']}")
    print(f"  Centroide = {obj['centroide']}")
    print(f"  Caja = {obj['caja']}")
    print()


# VISUALIZACIÓN DE RESULTADOS

plt.figure(figsize=(18, 12))

plt.subplot(3, 4, 1)
plt.imshow(imagen_rgb)
plt.title("Imagen original")
plt.axis("off")

plt.subplot(3, 4, 2)
plt.imshow(imagen_gris, cmap="gray")
plt.title("Escala de grises")
plt.axis("off")

plt.subplot(3, 4, 3)
plt.imshow(imagen_negativa, cmap="gray")
plt.title("Transformación negativa")
plt.axis("off")

plt.subplot(3, 4, 4)
plt.imshow(np.clip(np.abs(gradiente_x), 0, 255), cmap="gray")
plt.title("Sobel en X")
plt.axis("off")

plt.subplot(3, 4, 5)
plt.imshow(np.clip(np.abs(gradiente_y), 0, 255), cmap="gray")
plt.title("Sobel en Y")
plt.axis("off")

plt.subplot(3, 4, 6)
plt.imshow(bordes, cmap="gray")
plt.title("Magnitud del gradiente")
plt.axis("off")

plt.subplot(3, 4, 7)
plt.imshow(imagen_binaria, cmap="gray")
plt.title(f"Umbralización (T={umbral})")
plt.axis("off")

plt.subplot(3, 4, 8)
plt.imshow(etiquetas_filtradas, cmap="nipy_spectral")
plt.title(f"Objetos filtrados: {numero_objetos_filtrados}")
plt.axis("off")

plt.subplot(3, 4, 9)
plt.imshow(imagen_rgb)
plt.title("Centroides de objetos")
plt.axis("off")

for obj in propiedades:
    cy, cx = obj["centroide"]
    plt.plot(cx, cy, 'ro')
    plt.text(cx + 5, cy, str(obj["etiqueta"]), color='yellow', fontsize=12)

plt.subplot(3, 4, 10)
plt.imshow(harris_visual, cmap="hot")
plt.title("Respuesta Harris normalizada")
plt.axis("off")

plt.subplot(3, 4, 11)
plt.imshow(imagen_rgb)
plt.title("Esquinas detectadas")
plt.axis("off")

for y, x in esquinas:
    plt.plot(x, y, 'c.', markersize=4)

plt.tight_layout()
plt.show()