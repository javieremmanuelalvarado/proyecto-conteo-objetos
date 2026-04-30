# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 00:50:33 2026

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
    etiquetar_componentes_conectados,
    contar_objetos,
    eliminar_objetos_pequenos,
    calcular_propiedades
)

ruta = "imagenes_prueba/ejemplo_15.jpg"

# -----------------------------
# Lectura y preprocesamiento
# -----------------------------
imagen = leer_imagen(ruta)
imagen_gris = convertirBGR_a_grises(imagen)
imagen_negativa = transformacion_negativa(imagen_gris)
imagen_rgb = bgr_a_rgb(imagen)

kernel_promedio = filtro_promedio()
imagen_suavizada = aplicar_convolucion(imagen_gris, kernel_promedio)


respuesta_harris = calcular_matriz_harris(imagen_suavizada, k=0.04)
esquinas = detectar_esquinas_harris(respuesta_harris, umbral_relativo=0.05)

print("Número de esquinas detectadas:", len(esquinas))

# -----------------------------
# Detección de bordes con Sobel
# -----------------------------
sobel_x = kernel_sobel_x()
sobel_y = kernel_sobel_y()

gradiente_x = aplicar_convolucion(imagen_suavizada, sobel_x, aplicar_clip=False)
gradiente_y = aplicar_convolucion(imagen_suavizada, sobel_y, aplicar_clip=False)

bordes = calcular_magnitud_gradiente(gradiente_x, gradiente_y)

# -----------------------------
# Umbralización
# -----------------------------
umbral = 65
imagen_binaria = umbralizar_imagen(imagen_suavizada, umbral)

# Si alguna vez los objetos quedan negros y el fondo blanco, descomenta esto:
#imagen_binaria = 255 - imagen_binaria

# -----------------------------
# Segmentación y conteo
# -----------------------------
etiquetas, numero_objetos = etiquetar_componentes_conectados(imagen_binaria)

# Filtrado de objetos pequeños
min_area = 50
etiquetas_filtradas, numero_objetos_filtrados = eliminar_objetos_pequenos(etiquetas, min_area)

# Propiedades de cada objeto
propiedades = calcular_propiedades(etiquetas_filtradas)

# -----------------------------
# Resultados en consola
# -----------------------------
print("Número de objetos detectados antes del filtrado:", numero_objetos)
print("Número de objetos detectados después del filtrado:", numero_objetos_filtrados)
print()

for obj in propiedades:
    print(f"Objeto {obj['etiqueta']}:")
    print(f"  Área = {obj['area']}")
    print(f"  Centroide = {obj['centroide']}")
    print(f"  Caja = {obj['caja']}")
    print()

# -----------------------------
# Visualización
# -----------------------------
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
plt.title(f"Umbral (T={umbral})")
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
plt.imshow(respuesta_harris, cmap="hot")
plt.title("Respuesta Harris")
plt.axis("off")

plt.subplot(3, 4, 11)
plt.imshow(imagen_rgb)
plt.title("Esquinas detectadas")
plt.axis("off")

for y, x in esquinas:
    plt.plot(x, y, 'c.', markersize=4)

plt.tight_layout()
plt.show()