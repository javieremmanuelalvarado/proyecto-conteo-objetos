# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:29:08 2026

@author: emman
"""

import numpy as np

def umbralizar_imagen(imagen, umbral):
    """
    Convierte una imagen en escala de grises a binaria usando un umbral fijo.
    
    Parámetros:
    - imagen: imagen 2D en escala de grises
    - umbral: valor de corte
    
    Salida:
    - imagen binaria con valores 0 y 255
    """
    binaria = np.zeros_like(imagen, dtype=np.uint8)

    alto, ancho = imagen.shape

    for i in range(alto):
        for j in range(ancho):
            if imagen[i, j] >= umbral:
                binaria[i, j] = 255
            else:
                binaria[i, j] = 0

    return binaria


def calcular_umbral_otsu(imagen):
    """
    Calcula automáticamente el umbral óptimo usando el método de Otsu.

    Parámetro:
    - imagen: imagen en escala de grises

    Devuelve:
    - mejor_umbral: valor de umbral calculado
    """

    histograma = np.zeros(256)
    alto, ancho = imagen.shape

    # Calcular histograma manualmente
    for i in range(alto):
        for j in range(ancho):
            intensidad = int(imagen[i, j])
            histograma[intensidad] += 1

    total_pixeles = alto * ancho
    histograma = histograma / total_pixeles

    media_total = 0
    for i in range(256):
        media_total += i * histograma[i]

    peso_fondo = 0
    media_fondo = 0

    mejor_umbral = 0
    maxima_varianza = 0

    for t in range(256):
        peso_fondo += histograma[t]

        if peso_fondo == 0:
            continue

        peso_objeto = 1 - peso_fondo

        if peso_objeto == 0:
            break

        media_fondo += t * histograma[t]

        media_f = media_fondo / peso_fondo
        media_o = (media_total - media_fondo) / peso_objeto

        varianza_entre_clases = peso_fondo * peso_objeto * (media_f - media_o) ** 2

        if varianza_entre_clases > maxima_varianza:
            maxima_varianza = varianza_entre_clases
            mejor_umbral = t

    return mejor_umbral


def obtener_vecinos(i, j, alto, ancho):
    """
    Devuelve los vecinos válidos de un pixel usando conectividad de 8.
    """
    vecinos = []

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue

            ni = i + di
            nj = j + dj

            if 0 <= ni < alto and 0 <= nj < ancho:
                vecinos.append((ni, nj))

    return vecinos


def etiquetar_componentes_conectados(imagen_binaria):
    """
    Etiqueta manualmente las regiones conectadas en una imagen binaria.
    
    Considera como objeto los pixeles con valor 255.
    Usa conectividad de 8.
    
    Devuelve:
    - etiquetas: matriz con la etiqueta de cada región
    - num_etiquetas: número total de objetos encontrados
    """
    alto, ancho = imagen_binaria.shape

    etiquetas = np.zeros((alto, ancho), dtype=np.int32)
    etiqueta_actual = 0

    for i in range(alto):
        for j in range(ancho):
            if imagen_binaria[i, j] == 255 and etiquetas[i, j] == 0:
                etiqueta_actual += 1

                cola = [(i, j)]
                etiquetas[i, j] = etiqueta_actual

                while len(cola) > 0:
                    x, y = cola.pop(0)

                    vecinos = obtener_vecinos(x, y, alto, ancho)

                    for nx, ny in vecinos:
                        if imagen_binaria[nx, ny] == 255 and etiquetas[nx, ny] == 0:
                            etiquetas[nx, ny] = etiqueta_actual
                            cola.append((nx, ny))

    return etiquetas, etiqueta_actual


def contar_objetos(imagen_binaria):
    """
    Cuenta los objetos en una imagen binaria usando componentes conectados.
    """
    _, cantidad = etiquetar_componentes_conectados(imagen_binaria)
    return cantidad


def eliminar_objetos_pequenos(etiquetas, min_area):
    """
    Elimina regiones cuya área sea menor que min_area.
    
    Parámetros:
    - etiquetas: imagen etiquetada
    - min_area: área mínima permitida
    
    Devuelve:
    - etiquetas_filtradas: imagen etiquetada sin objetos pequeños
    - numero_objetos: número final de objetos válidos
    """
    etiquetas_filtradas = np.copy(etiquetas)
    etiquetas_unicas = np.unique(etiquetas_filtradas)

    # Quitamos el fondo (etiqueta 0)
    etiquetas_unicas = etiquetas_unicas[etiquetas_unicas != 0]

    for etiqueta in etiquetas_unicas:
        area = np.sum(etiquetas_filtradas == etiqueta)

        if area < min_area:
            etiquetas_filtradas[etiquetas_filtradas == etiqueta] = 0

    # Re-etiquetar para que queden consecutivas: 1, 2, 3, ...
    etiquetas_finales = np.zeros_like(etiquetas_filtradas, dtype=np.int32)
    nueva_etiqueta = 0

    etiquetas_unicas = np.unique(etiquetas_filtradas)
    etiquetas_unicas = etiquetas_unicas[etiquetas_unicas != 0]

    for etiqueta in etiquetas_unicas:
        nueva_etiqueta += 1
        etiquetas_finales[etiquetas_filtradas == etiqueta] = nueva_etiqueta

    return etiquetas_finales, nueva_etiqueta


def calcular_propiedades(etiquetas):
    """
    Calcula propiedades básicas de cada objeto etiquetado.
    
    Propiedades calculadas:
    - área
    - centroide
    - caja delimitadora
    
    Devuelve:
    - lista de diccionarios con propiedades por objeto
    """
    propiedades = []
    etiquetas_unicas = np.unique(etiquetas)

    # Quitamos fondo
    etiquetas_unicas = etiquetas_unicas[etiquetas_unicas != 0]

    for etiqueta in etiquetas_unicas:
        coords = np.where(etiquetas == etiqueta)

        filas = coords[0]
        columnas = coords[1]

        area = len(filas)

        centroide_y = np.mean(filas)
        centroide_x = np.mean(columnas)

        min_fila = np.min(filas)
        max_fila = np.max(filas)
        min_col = np.min(columnas)
        max_col = np.max(columnas)

        propiedades.append({
            "etiqueta": int(etiqueta),
            "area": int(area),
            "centroide": (float(centroide_y), float(centroide_x)),
            "caja": (int(min_fila), int(min_col), int(max_fila), int(max_col))
        })

    return propiedades